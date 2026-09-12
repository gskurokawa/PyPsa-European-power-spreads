"""Close the flow-based constraints, and check them against what really bound.

    python scripts/31_close_domain.py --inspect     # what net-position data exists
    python scripts/31_close_domain.py               # the full test

THE TEST
--------
Every flow-based constraint says

    SUM over ALL zones of  PTDF(zone) x NetPosition(zone)   <=   RAM

Put the OBSERVED net positions into the left-hand side and the margin

    margin = RAM - SUM PTDF x NP_observed

should behave in a very particular way: for the constraints JAO reports as
binding in that hour it should be about ZERO, and for everything else it
should be positive.  That is a complete end-to-end check of the parse, the
hub mapping, the hour alignment, the sign conventions and the fixed-term
arithmetic - and it needs no dispatch model and no LP at all.

It is also what settles the ALEGrO orientation.  The domain splits that one
HVDC into hub_ALBE and hub_ALDE, and which end takes which sign is a
convention: this tries both and keeps whichever puts the binding constraints
nearer zero.

WHAT IT NEEDS
-------------
Hourly net positions for all twelve Core zones plus ALEGrO.  Eight come from
this repo's own exchange data; five (HR, HU, RO, SI, SK) and ALEGrO come from
script 30.  Until those exist the margin is computed from what is available
and the script reports the size of the terms it had to leave out, so a partial
answer is never mistaken for a complete one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW  = ROOT / "data" / "raw"

CANDIDATES = ["boundary_position.parquet", "exchange.parquet",
              "commercial_flows.parquet", "net_flows.parquet"]
MODEL_ZONES   = ["DE_LU", "FR", "PL", "NL", "BE", "AT", "CH", "CZ"]
OMITTED_CORE  = ["HR", "HU", "RO", "SI", "SK"]


def inspect() -> None:
    print("=" * 80)
    print("WHAT NET-POSITION DATA IS ALREADY IN THE REPO")
    print("=" * 80)
    for name in CANDIDATES:
        p = PROC / name
        if not p.exists():
            print(f"\n  {name:<28} not present")
            continue
        d = pd.read_parquet(p)
        print(f"\n  {name}")
        print(f"    shape {d.shape}   index {type(d.index).__name__}")
        if isinstance(d.index, pd.DatetimeIndex):
            print(f"    {d.index.min()} .. {d.index.max()}  tz={d.index.tz}")
        print(f"    columns ({len(d.columns)}): "
              f"{', '.join(map(str, d.columns[:14]))}"
              f"{' ...' if len(d.columns) > 14 else ''}")
        num = d.select_dtypes("number")
        if len(num.columns):
            print(f"    means: "
                  + ", ".join(f"{c}={num[c].mean():.0f}"
                              for c in num.columns[:8]))

    print("\n" + "=" * 80)
    print("WHAT SCRIPT 30 HAS FETCHED")
    print("=" * 80)
    npdir = RAW / "net_position"
    if npdir.exists():
        files = sorted(npdir.glob("*.parquet"))
        by_zone = {}
        for f in files:
            by_zone.setdefault(f.stem.rsplit("_", 1)[0], []).append(f)
        for z, fs in sorted(by_zone.items()):
            print(f"  {z:<10} {len(fs):>3} monthly files")
        missing = [z for z in OMITTED_CORE if z not in by_zone]
        if missing:
            print(f"  still missing: {missing}")
    else:
        print("  data/raw/net_position/ does not exist yet - run script 30")
        print("  once the ENTSO-E platform is answering again.")

    scdir = RAW / "scheduled"
    if scdir.exists():
        be_de = [f.name for f in scdir.glob("*") if "BE" in f.name
                 and "DE" in f.name]
        print(f"\n  BE-DE scheduled exchange files (ALEGrO): {len(be_de)}")

    print("\n" + "=" * 80)
    print("NEXT")
    print("=" * 80)
    print("  Paste the above.  The column layout decides how the eight")
    print("  modelled zones' net positions are assembled; everything after")
    print("  that is the same regardless of which file they come from.")


def _np_total(zones):
    """Net position across ALL borders, from net_flows.parquet.

    boundary_position.parquet sums only the thirteen modelled borders, so
    France reads about +5,100 MW where its true average net export in 2024
    was nearer +10 GW - the difference being Spain, Italy and Britain.  JAO's
    net position might be either quantity; this builds the other one so both
    can be tested rather than argued about.
    """
    nf = pd.read_parquet(PROC / "net_flows.parquet")
    out = pd.DataFrame(index=nf.index)
    for z in zones:
        tot = pd.Series(0.0, index=nf.index)
        for c in nf.columns:
            if ">" not in str(c):
                continue
            a, b = str(c).split(">")
            if a == z:
                tot = tot + nf[c].fillna(0.0)      # exports count positive
            elif b == z:
                tot = tot - nf[c].fillna(0.0)
        out[z] = tot
    return out


def full_test() -> None:
    dom = pd.read_parquet(PROC / "fb_domain.parquet")
    dom["t"] = pd.to_datetime(dom["t"], utc=True)
    print(f"  domain: {len(dom):,} constraints, {dom['t'].nunique():,} hours")

    # ---------------- the two candidate net-position definitions ----------
    bp = pd.read_parquet(PROC / "boundary_position.parquet")
    defs = {"boundary (modelled borders only)": bp}
    try:
        defs["total (all borders)"] = _np_total(MODEL_ZONES)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  could not build the all-borders version: {str(exc)[:70]}")

    # ---------------- ALEGrO, from data already on disk -------------------
    ex = pd.read_parquet(PROC / "exchange.parquet")
    alegro = None
    for c in ("DE_LU>BE", "BE>DE_LU"):
        if c in ex.columns:
            alegro = ex[c].fillna(0.0) * (1 if c == "DE_LU>BE" else -1)
            print(f"  ALEGrO flow from exchange.parquet[{c}], "
                  f"mean {alegro.mean():.0f} MW")
            break

    # ---------------- which constraints can be judged without HR..SK ------
    om = [c for c in dom.columns if c.startswith("fixed_")]
    dom["omitted_mw"] = dom[om].abs().sum(axis=1) * 4000
    clean = dom["omitted_mw"] < 50
    print(f"\n  constraints whose omitted-Core terms are worth <50 MW: "
          f"{int(clean.sum()):,} of {len(dom):,} ({100*clean.mean():.1f} %)")
    print("  those can be judged today; the rest wait for script 30.")

    # ---------------- label what actually bound ---------------------------
    act = pd.read_csv(RAW / "jao" / "activeFbConstraints_2024.csv",
                      low_memory=False)
    act.columns = [str(c).strip().lstrip("\ufeff").lower() for c in act.columns]
    act["t"] = pd.to_datetime(act["datetimeutc"], utc=True, errors="coerce")
    act = act[act["shadowprice"].abs() > 1e-9]
    act["cid"] = (act["cneceic"].astype(str) + "|" + act["brancheic"].astype(str)
                  + "|" + act["direction"].astype(str))
    bound = set(act["t"].astype("int64").astype(str) + "|" + act["cid"])
    dom["bound"] = (dom["t"].astype("int64").astype(str) + "|"
                    + dom["cid"]).isin(bound)
    sub = dom[clean].copy()
    print(f"  of those, {int(sub['bound'].sum()):,} actually bound")
    if sub["bound"].sum() < 30:
        print("  too few to judge - let script 22 fetch more days first")
        return

    # ---------------- score every combination -----------------------------
    print("\n" + "=" * 82)
    print("MARGIN = RAM - SUM(PTDF x observed NP), BY DEFINITION AND ALEGRO SIGN")
    print("=" * 82)
    print("  A binding constraint should sit at margin ~0.  A slack one should")
    print("  be positive.  The right combination separates them; a wrong one")
    print("  will not.\n")
    print(f"  {'net position':<34}{'ALEGrO':>8}{'bound med':>11}"
          f"{'slack med':>11}{'|bound|<50':>12}{'separation':>12}")
    print("  (a coverage line prints first for each definition - if that is")
    print("   not ~100 %, the numbers under it mean nothing)")

    best = None
    for label, np_df in defs.items():
        # Align on the tz-AWARE index.  sub["t"].values strips the timezone
        # to naive datetime64, which matches nothing in a tz-aware
        # DatetimeIndex, and every lookup comes back NaN - silently, because
        # nan_to_num then turns it into a zero contribution.  That produced a
        # table where all six combinations were identical.
        idx = pd.DatetimeIndex(sub["t"])
        npx = np_df.copy()
        if npx.index.tz is None:
            npx.index = npx.index.tz_localize("UTC")
        npx = npx.reindex(idx)
        cover = 100 * npx.notna().all(axis=1).mean()
        print(f"    [{label}] net positions matched for {cover:.1f} % of rows")
        if cover < 50:
            print("     ALIGNMENT FAILED - not scoring this definition")
            continue
        lhs = pd.Series(0.0, index=range(len(sub)))
        for z in MODEL_ZONES:
            col = f"ptdf_{z}"
            if col in sub.columns and z in npx.columns:
                lhs = lhs + sub[col].values * np.nan_to_num(npx[z].values)
        for sgn in (0, +1, -1):
            add = pd.Series(0.0, index=range(len(sub)))
            if sgn and alegro is not None:
                al = alegro.copy()
                if al.index.tz is None:
                    al.index = al.index.tz_localize("UTC")
                a = al.reindex(idx).fillna(0.0).values
                for hub, mult in (("fixedhub_albe", +1), ("fixedhub_alde", -1)):
                    if hub in sub.columns:
                        add = add + sub[hub].values * (sgn * mult * a)
            margin = sub["ram"].values - (lhs + add).values
            b = margin[sub["bound"].values]
            s = margin[~sub["bound"].values]
            hit = 100 * np.mean(np.abs(b) < 50)
            sep = np.median(s) - np.median(b)
            tag = {0: "off", 1: "+", -1: "-"}[sgn]
            print(f"  {label:<34}{tag:>8}{np.median(b):>11.0f}"
                  f"{np.median(s):>11.0f}{hit:>11.0f}%{sep:>12.0f}")
            if best is None or hit > best[0]:
                best = (hit, label, tag, sep)

    print(f"\n  best: {best[1]}, ALEGrO {best[2]} - "
          f"{best[0]:.0f}% of binding constraints within 50 MW of zero, "
          f"separation {best[3]:.0f} MW")
    print("\n  A high share near zero AND a clear positive separation means the")
    print("  whole pipeline reproduces the real constraint set.  A median")
    print("  binding margin far from zero means a term is still missing or a")
    print("  sign is wrong - and the two definitions above tell you which.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    inspect() if a.inspect else full_test()
