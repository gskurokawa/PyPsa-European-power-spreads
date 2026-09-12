"""Does the published flow-based domain for a given year check out?

    python scripts/42_validate_domain_year.py --year 2024      # the control
    python scripts/42_validate_domain_year.py --year 2025

Script 32 searched for the right pair of conventions and found one:

    published ram,  sign +,  Switzerland left out,  JAO's own netPos

On 2024 that put the binding constraints on a median margin of 0 MW, with 47%
inside 50 MW and a separation of 880 MW between binding and slack constraints.
That search does not need repeating. What DOES need repeating, once per year of
data used, is the CHECK: apply the settled convention to a year and confirm the
domain reproduces the constraint set JAO says was binding.

FINDINGS 12.6 ran that check on 2024 only. This script runs the same check on
any year, so a CNEC result reported on 2025 rests on a validated 2025 domain
rather than on the assumption that a second year downloaded the same way is
equally sound.

Three things are reported, in order of how badly they would hurt.

  COVERAGE  - how many hours of the year the domain actually contains, and how
              many carry a full set of Core net positions. A silently missing
              day is the failure mode script 22 was rewritten to catch, and it
              would show up here as hours the model solves with NO flow-based
              constraint at all - unconstrained trade, not an error.
  SANITY    - NaNs, negative RAM, PTDF magnitudes, constraints per hour. Cheap,
              and catches a malformed parse before it reaches the LP.
  MARGIN    - the real test. For every constraint JAO published as binding,
              RAM - PTDF x netPos should be near zero; for slack ones it should
              be clearly positive. If the binding median drifts off zero, the
              domain for that year is not describing the same thing 2024's did.

A year that fails MARGIN must not be used for a reported result. A year that
fails only COVERAGE can be used with the gap stated, since the model simply
runs unconstrained in those hours - but say so, because unconstrained hours
bias the spreads DOWN and that flatters the model.
"""
from __future__ import annotations

import argparse
import builtins
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

MODEL_ZONES = ["DE_LU", "FR", "PL", "NL", "BE", "AT", "CH", "CZ"]

# fb_domain column -> the netPos column carrying that zone. Settled in
# script 32; the five fixed_ zones are Core members the model does not
# represent, and they still load the constraints.
COL = {"ptdf_DE_LU": "hub_de", "ptdf_FR": "hub_fr", "ptdf_PL": "hub_pl",
       "ptdf_NL": "hub_nl", "ptdf_BE": "hub_be", "ptdf_AT": "hub_at",
       "ptdf_CZ": "hub_cz",
       "fixed_HR": "hub_hr", "fixed_HU": "hub_hu", "fixed_RO": "hub_ro",
       "fixed_SI": "hub_si", "fixed_SK": "hub_sk"}


def norm(df):
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿").lower() for c in df.columns]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--omitted-mw", type=float, default=50.0,
                    help="drop constraints whose non-modelled Core zones could "
                         "move the LHS by more than this, as script 32 did. "
                         "Set high to judge every constraint.")
    args = ap.parse_args()
    y = args.year

    # Tee to a file. The 2024 numbers here are the reference every later year
    # is judged against, so they need to be on disk, not in a scrollback.
    logf = (ROOT / "logs" / f"domain_validate_{y}.txt").open(
        "w", encoding="utf-8")

    # Read the real print off builtins, not off the enclosing scope: naming a
    # local `print` below makes `print` local for the WHOLE function, so a
    # plain `_print = print` here is an unbound read, not a capture.
    _print = builtins.print

    def print(*a, **k):                                       # noqa: A001
        _print(*a, **k)
        logf.write(" ".join(str(x) for x in a) + "\n")

    dom_path = PROC / f"fb_domain_{y}.parquet"
    act_path = RAW / "jao" / f"activeFbConstraints_{y}.csv"
    np_path = RAW / "jao" / f"netPos_{y}.csv"
    for p in (dom_path, act_path, np_path):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    # ---------------------------------------------------------------- coverage
    print("=" * 88)
    print(f"COVERAGE  {y}")
    print("=" * 88)

    dom = pd.read_parquet(dom_path)
    dom["t"] = pd.to_datetime(dom["t"], utc=True)

    grid = pd.date_range(f"{y}-01-01", f"{y + 1}-01-01", freq="h",
                         tz="UTC", inclusive="left")
    have_h = pd.DatetimeIndex(dom["t"].unique()).sort_values()
    missing = grid.difference(have_h)

    print(f"  domain rows                {len(dom):>12,}")
    print(f"  hours in the year          {len(grid):>12,}")
    print(f"  hours present in domain    {len(have_h):>12,}")
    print(f"  hours MISSING              {len(missing):>12,}"
          f"   ({100 * len(missing) / len(grid):.2f} %)")
    if len(missing):
        gaps, start, prev = [], missing[0], missing[0]
        for t in missing[1:]:
            if (t - prev) > pd.Timedelta("1h"):
                gaps.append((start, prev))
                start = t
            prev = t
        gaps.append((start, prev))
        print(f"  {len(gaps)} contiguous gap(s):")
        for a, b in gaps[:20]:
            n = int((b - a) / pd.Timedelta("1h")) + 1
            print(f"    {a:%Y-%m-%d %H:%M} .. {b:%Y-%m-%d %H:%M}  ({n} h)")
        if len(gaps) > 20:
            print(f"    ... and {len(gaps) - 20} more")
        print("\n  In a missing hour the model has NO flow-based constraint.")
        print("  It trades freely, so spreads in those hours are too small.")

    per_hour = dom.groupby("t").size()
    print(f"\n  constraints per hour       "
          f"min {per_hour.min()}, median {per_hour.median():.0f}, "
          f"max {per_hour.max()}")

    # ------------------------------------------------------------------ sanity
    print("\n" + "=" * 88)
    print(f"SANITY  {y}")
    print("=" * 88)
    ptdf_cols = [c for c in dom.columns if c.startswith("ptdf_")]
    print(f"  ptdf columns               {len(ptdf_cols)}  "
          f"({', '.join(c[5:] for c in ptdf_cols)})")
    print(f"  ram: mean {dom['ram'].mean():.0f} MW, "
          f"min {dom['ram'].min():.0f}, max {dom['ram'].max():.0f}")
    print(f"  ram NaN                    {int(dom['ram'].isna().sum()):,}")
    print(f"  ram < 0                    {int((dom['ram'] < 0).sum()):,}"
          f"   ({100 * (dom['ram'] < 0).mean():.2f} %)")
    print(f"  |ptdf| > 1                 "
          f"{int((dom[ptdf_cols].abs() > 1).sum().sum()):,}")
    print(f"  ptdf NaN                   "
          f"{int(dom[ptdf_cols].isna().sum().sum()):,}")
    print("\n  A negative RAM is not a bug - it means the element was already")
    print("  overloaded in the reference case - but a large share of them")
    print("  would make the domain infeasible before the model does anything.")

    # ------------------------------------------------------------------ margin
    print("\n" + "=" * 88)
    print(f"MARGIN  {y}   published ram, sign +, CH excluded  (settled in 32)")
    print("=" * 88)

    om = [c for c in dom.columns if c.startswith("fixed_")]
    if om:
        dom["omitted_mw"] = dom[om].abs().sum(axis=1) * 4000
        sub = dom[dom["omitted_mw"] < args.omitted_mw].copy()
    else:
        sub = dom.copy()
    print(f"  judging {len(sub):,} of {len(dom):,} constraints "
          f"(omitted-zone effect < {args.omitted_mw:.0f} MW)")

    act = norm(pd.read_csv(act_path, low_memory=False))
    act["t"] = pd.to_datetime(act["datetimeutc"], utc=True, errors="coerce")
    act = act[act["shadowprice"].abs() > 1e-9]
    act["cid"] = (act["cneceic"].astype(str) + "|"
                  + act["brancheic"].astype(str) + "|"
                  + act["direction"].astype(str))
    bound = set(act["t"].astype("int64").astype(str) + "|" + act["cid"])
    sub["bound"] = (sub["t"].astype("int64").astype(str) + "|"
                    + sub["cid"]).isin(bound)
    print(f"  JAO published {len(act):,} binding constraint-hours; "
          f"{int(sub['bound'].sum()):,} of them are in the judged set")

    jn = norm(pd.read_csv(np_path, low_memory=False))
    jn["t"] = pd.to_datetime(jn["datetimeutc"], utc=True, errors="coerce")
    jn = jn.set_index("t")
    jn = jn[~jn.index.duplicated(keep="first")]
    print(f"  netPos: {len(jn):,} hours, {jn.index.min()} .. {jn.index.max()}")

    have = {k: v for k, v in COL.items()
            if k in sub.columns and v in jn.columns}
    print(f"  matched {len(have)} of 12 Core zones")
    if len(have) < 12:
        print(f"  NOT matched: {[k for k in COL if k not in have]}")

    idx = pd.DatetimeIndex(sub["t"])
    npx = jn.reindex(idx)
    cover = 100 * npx[list(have.values())].notna().all(axis=1).mean()
    print(f"  constraint-rows with a full set of net positions: {cover:.1f} %")

    lhs = np.zeros(len(sub))
    for pcol, ncol in have.items():
        lhs = lhs + sub[pcol].values * np.nan_to_num(npx[ncol].values)
    margin = sub["ram"].values - lhs

    b = margin[sub["bound"].values]
    s = margin[~sub["bound"].values]
    b, s = b[np.isfinite(b)], s[np.isfinite(s)]

    print(f"\n  binding constraints   n {len(b):,}")
    print(f"    median margin            {np.median(b):>10.0f} MW"
          f"      (2024 reference: 0)")
    print(f"    within 50 MW of zero     {100 * np.mean(np.abs(b) < 50):>9.0f} %"
          f"      (2024 reference: 47%)")
    print(f"    within 100 MW of zero    {100 * np.mean(np.abs(b) < 100):>9.0f} %"
          f"      (2024 reference: 58%)")
    print(f"  slack constraints     n {len(s):,}")
    print(f"    median margin            {np.median(s):>10.0f} MW")
    print(f"  separation (slack - bound) "
          f"{np.median(s) - np.median(b):>10.0f} MW"
          f"      (2024 reference: 880)")

    print("\n  VERDICT")
    ok_med = abs(np.median(b)) < 100
    ok_hit = 100 * np.mean(np.abs(b) < 50) > 35
    ok_sep = (np.median(s) - np.median(b)) > 400
    for label, ok in (("binding median near zero", ok_med),
                      ("hit rate comparable to 2024", ok_hit),
                      ("clear binding/slack separation", ok_sep)):
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")
    if ok_med and ok_hit and ok_sep:
        print(f"\n  The {y} domain reproduces JAO's own constraint set under the")
        print("  convention settled on 2024. It is sound enough to report on.")
    else:
        print(f"\n  The {y} domain does NOT behave like 2024's. Do not report a")
        print("  CNEC result on it until the difference is explained.")

    logf.close()
    _print(f"\nwritten to logs/domain_validate_{y}.txt")


if __name__ == "__main__":
    main()
