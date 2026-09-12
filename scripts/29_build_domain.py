"""Turn the raw JAO domain into the constraint set the model will consume.

    python scripts/29_build_domain.py
    python scripts/29_build_domain.py --zones DE_LU FR PL     # 3-zone cell

Reads data/raw/jao/domain2024/, writes data/processed/fb_domain.parquet.
Works on whatever chunks exist - it does not need the download finished.

WHAT A FLOW-BASED CONSTRAINT IS, IN THIS MODEL'S TERMS
-----------------------------------------------------
For each critical network element under each contingency, in each hour:

    SUM over all zones of  PTDF(zone) x NetPosition(zone)   <=   RAM

The sum runs over ALL twelve Core zones plus a set of virtual hubs, not just
the ones this model represents. That is the whole difficulty. A zone this
model does not solve for still loads the wire, so its term cannot be dropped -
it has to be moved to the other side of the inequality using what that zone
actually did:

    SUM over MODELLED zones of PTDF x NP  <=  RAM  -  SUM over FIXED terms

The right-hand side is then a number per constraint per hour, and the left is
a linear expression in the model's own link flows. That is exactly the shape
linopy wants.

So this script does three things: assemble the constraints, work out which
fixed terms actually matter (they are not all worth chasing data for), and
write both sides out.

THE FIXED TERMS, AND WHY THEY ARE NOT ALL EQUAL
-----------------------------------------------
Two kinds:

  Core zones this model omits - Croatia, Hungary, Romania, Slovenia,
  Slovakia. Real bidding zones with real net positions, several GW.

  Virtual hubs - ALEGrO (the BE-DE HVDC, as hub_ALBE / hub_ALDE) and nine
  external-border hubs such as DE-DK1, NorNed, SwePol. These are single
  interconnectors, usually a few hundred MW, and their PTDFs are often zero.

A term matters in proportion to |PTDF| x (how much power flows through it).
RAM averages 844 MW, so a term worth 5 MW is noise and a term worth 300 MW is
not. This script sizes every one of them so the data-collection effort goes
where it changes the answer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JAO  = ROOT / "data" / "raw" / "jao"
DOM  = JAO / "domain2024"      # overridden by --domain-dir
OUT  = ROOT / "data" / "processed"

# this model's zones -> the domain's PTDF column for each
MODEL_ZONE_PTDF = {
    "DE_LU": "ptdf_de", "FR": "ptdf_fr", "PL": "ptdf_pl", "NL": "ptdf_nl",
    "BE":    "ptdf_be", "AT": "ptdf_at", "CH": "ptdf_ch", "CZ": "ptdf_cz",
}
# Core zones this model does not represent: fixed to observed net position
OMITTED_CORE = {"HR": "ptdf_hr", "HU": "ptdf_hu", "RO": "ptdf_ro",
                "SI": "ptdf_si", "SK": "ptdf_sk"}

# nominal size of each virtual hub, MW, used only to SIZE its contribution.
# ALEGrO 1000 MW; the external hubs are their interconnectors' ratings.
HUB_SCALE = {
    "ptdf_albe": 1000, "ptdf_alde": 1000,
    "ptdf_de_dk1_vh": 1780, "ptdf_de_dk2_bighub": 600,
    "ptdf_de_no2_bighub": 1400, "ptdf_de_se4_baltic": 615,
    "ptdf_nl_no2_norned": 700, "ptdf_nl_dk1_cobra": 700,
    "ptdf_pl_se4_swepol": 600, "ptdf_pl_lt_bighub": 500,
    "ptdf_ro_bg_vh": 1000,
}
# a Core zone's net position swings by GW, not MW
ZONE_SCALE = 4000


def norm(df):
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿").lower() for c in df.columns]
    return df


def load_domain() -> pd.DataFrame:
    files = sorted(DOM.glob("fc_*.csv.gz"))
    if not files:
        raise SystemExit("no domain chunks yet - let script 22 run first")
    frames, skipped = [], 0
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception:                                     # noqa: BLE001
            skipped += 1
    d = norm(pd.concat(frames, ignore_index=True))
    d["t"] = pd.to_datetime(d["datetimeutc"], utc=True, errors="coerce")
    print(f"  {len(files) - skipped} chunks, {len(d):,} rows, "
          f"{d['t'].nunique():,} hours "
          f"({d['t'].min():%Y-%m-%d} .. {d['t'].max():%Y-%m-%d})")
    if skipped:
        print(f"  ({skipped} chunk(s) skipped - still being written)")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", nargs="*", default=list(MODEL_ZONE_PTDF),
                    help="which zones stay endogenous (the 2x2 scope axis)")
    ap.add_argument("--out", default="fb_domain.parquet")
    ap.add_argument("--domain-dir", default="domain2024",
                    help="which year's domain folder to read "
                         "(they are kept apart so a run cannot mix years)")
    args = ap.parse_args()

    global DOM
    DOM = JAO / args.domain_dir
    print("=" * 84)
    print(f"1  LOAD  ({args.domain_dir})")
    print("=" * 84)
    d = load_domain()

    # one row per constraint per hour; the unique key from script 28
    d["cid"] = (d["cneeic"].astype(str) + "|"
                + d["contbrancheic1"].astype(str) + "|"
                + d["direction"].astype(str))
    before = len(d)
    d = d.drop_duplicates(subset=["t", "cid"], keep="first")
    print(f"  {before - len(d):,} exact duplicate (hour, constraint) rows dropped")

    d["ram"] = pd.to_numeric(d["ram"], errors="coerce")
    d = d[d["ram"].notna()]
    ptdf_cols = [c for c in d.columns if c.startswith("ptdf_")]
    for c in ptdf_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    print("\n" + "=" * 84)
    print("2  HOW MUCH DOES EACH FIXED TERM ACTUALLY MATTER?")
    print("=" * 84)
    print("  contribution = |PTDF| x a typical flow, in MW of RAM consumed.")
    print(f"  For scale: RAM averages {d['ram'].mean():.0f} MW.\n")
    print(f"  {'term':<24}{'non-zero':>10}{'mean|PTDF|':>12}"
          f"{'p99|PTDF|':>11}{'scale':>8}{'typical MW':>12}")
    rows = []
    for label, col, scale in (
        [(f"{z} (Core, omitted)", c, ZONE_SCALE)
         for z, c in OMITTED_CORE.items()]
        + [(c.replace("ptdf_", ""), c, HUB_SCALE.get(c, 500))
           for c in ptdf_cols
           if c not in MODEL_ZONE_PTDF.values()
           and c not in OMITTED_CORE.values()]
    ):
        if col not in d.columns:
            continue
        v = d[col].abs()
        nz = (v > 1e-6).mean()
        typical = v.mean() * scale
        rows.append((label, col, nz, v.mean(), v.quantile(0.99), scale, typical))
    for label, col, nz, mean, p99, scale, typical in sorted(
            rows, key=lambda r: -r[6]):
        flag = "  <- matters" if typical > 25 else ""
        print(f"  {label:<24}{100*nz:>9.1f}%{mean:>12.4f}{p99:>11.4f}"
              f"{scale:>8}{typical:>12.1f}{flag}")

    print("\n  Terms above ~25 MW need real hourly data (script 30).")
    print("  Terms below it can be set to zero with a stated error bound.")

    print("\n" + "=" * 84)
    print("3  ASSEMBLE")
    print("=" * 84)
    keep_zones = [z for z in args.zones if z in MODEL_ZONE_PTDF]
    fixed_zones = [z for z in MODEL_ZONE_PTDF if z not in keep_zones]
    print(f"  endogenous: {keep_zones}")
    print(f"  fixed to observed: {fixed_zones + list(OMITTED_CORE)}")

    out = pd.DataFrame({
        "t":   d["t"].values,
        "cid": d["cid"].values,
        "ram": d["ram"].values,
        "cnename":  d["cnename"].values,
        "contname": d["contname"].values,
        "tso":      d["tso"].values,
        "direction": d["direction"].values,
    })
    for z in keep_zones:                       # left-hand side, the unknowns
        out[f"ptdf_{z}"] = d[MODEL_ZONE_PTDF[z]].values
    for z in fixed_zones:                      # right-hand side, once flows known
        out[f"fixed_{z}"] = d[MODEL_ZONE_PTDF[z]].values
    for z, c in OMITTED_CORE.items():
        out[f"fixed_{z}"] = d[c].values
    for c in ptdf_cols:
        if c not in MODEL_ZONE_PTDF.values() and c not in OMITTED_CORE.values():
            out[f"fixedhub_{c.replace('ptdf_', '')}"] = d[c].values

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / args.out
    out.sort_values(["t", "cid"]).to_parquet(path, index=False)

    per_hour = len(out) / max(out["t"].nunique(), 1)
    print(f"\n  wrote {path.name}: {len(out):,} rows, "
          f"{out['t'].nunique():,} hours, {per_hour:.0f} constraints/hour")
    print(f"  a 30-day chunk carries {per_hour*720:,.0f} constraints")
    print(f"  a full year carries    {per_hour*8784:,.0f}")

    print("\n" + "=" * 84)
    print("4  WHAT THIS DOES NOT YET HAVE")
    print("=" * 84)
    print("  Every fixed_* column is a PTDF, not yet a contribution: it still")
    print("  needs that zone's or hub's observed hourly net position to become")
    print("  a number of MW.  Until then the right-hand side is RAM alone,")
    print("  which OVERSTATES the room available to the market.  Script 30")
    print("  pulls those series; this file is the half that does not depend")
    print("  on them, and the columns are laid out so the join is one merge.")


if __name__ == "__main__":
    main()
