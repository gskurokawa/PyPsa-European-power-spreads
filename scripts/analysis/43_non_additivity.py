"""How much do per-border limits overstate what a zone can actually do?

    python scripts/analysis/43_non_additivity.py
    python scripts/analysis/43_non_additivity.py --date 2024-07-01 --zones DE FR PL

Section 4.1 argues that transmission capacity cannot be described border by
border, because every trade loads the same physical circuits. That argument is
usually made by quoting ENTSO-E's own warning that NTC values must not be
added. This measures it instead, from one publisher, for one hour at a time,
with nothing estimated.

JAO publishes two things for each hour of the flow-based domain:

  maxExchanges   the largest exchange possible across one border, computed on
                 the assumption that every other zone's net position is zero
  maxNetPos      the largest net position a zone may hold, across all of its
                 CORE borders at once

Add up a zone's border maxima and compare against its net-position maximum.
Both come from JAO, both describe the same hour, and the ratio is the factor by
which a set of independent per-border limits would overstate what that zone can
do simultaneously.

A ratio of 1.0 would mean the borders are independent and additive. Anything
above 1.0 is the size of the error a per-border representation makes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "jao"

# Each modelled zone's physical borders WITH OTHER CORE ZONES. Non-Core borders
# (Switzerland, Great Britain, the Nordics, Italy, Iberia) are excluded because
# maxNetPos is a CORE net position and does not cover them - the scope mismatch
# that corrupted an earlier version of the flow-based constraint.
CORE_BORDERS = {
    "DE": ["AT", "BE", "CZ", "FR", "NL", "PL"],
    "FR": ["BE", "DE"],
    "PL": ["CZ", "DE", "SK"],
    "NL": ["BE", "DE"],
    "BE": ["DE", "FR", "NL"],
    "AT": ["CZ", "DE", "HU", "SI"],
    "CZ": ["AT", "DE", "PL", "SK"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchanges", default=None,
                    help="maxExchanges csv; default is the sample on disk")
    ap.add_argument("--netpos", default=None,
                    help="maxNetPos csv; default is picked from the year")
    ap.add_argument("--zones", nargs="+", default=list(CORE_BORDERS))
    args = ap.parse_args()

    mx_path = Path(args.exchanges) if args.exchanges else next(
        iter(sorted(RAW.glob("maxExchanges_*.csv"))))
    mx = pd.read_csv(mx_path)
    mx["t"] = pd.to_datetime(mx["dateTimeUtc"], utc=True)
    year = int(mx["t"].dt.year.mode()[0])

    mn_path = Path(args.netpos) if args.netpos else RAW / f"maxNetPos_{year}.csv"
    mn = pd.read_csv(mn_path)
    mn["t"] = pd.to_datetime(mn["dateTimeUtc"], utc=True)
    mn = mn.set_index("t")

    hours = mx["t"]
    print(f"maxExchanges: {mx_path.name}, {len(mx)} hours, "
          f"{hours.min()} .. {hours.max()}")
    print(f"maxNetPos:    {mn_path.name}, {len(mn):,} hours")
    print()

    rows = []
    for zone in args.zones:
        neighbours = CORE_BORDERS.get(zone)
        if not neighbours:
            continue
        cols = [f"border_{zone}_{n}" for n in neighbours]
        missing = [c for c in cols if c not in mx.columns]
        if missing:
            print(f"  {zone}: missing columns {missing} - skipped")
            continue
        if f"max{zone}" not in mn.columns:
            print(f"  {zone}: no max{zone} in maxNetPos - skipped")
            continue

        border_sum = mx.set_index("t")[cols].sum(axis=1)
        net_max = mn.reindex(border_sum.index)[f"max{zone}"].astype(float)
        ratio = border_sum / net_max

        rows.append({
            "zone": zone,
            "borders": len(neighbours),
            "sum of border maxima, MW": round(border_sum.mean()),
            "max net position, MW": round(net_max.mean()),
            "overstatement": round(float(ratio.mean()), 2),
            "min": round(float(ratio.min()), 2),
            "max": round(float(ratio.max()), 2),
        })

    out = pd.DataFrame(rows).set_index("zone")
    print("=" * 92)
    print(f"PER-BORDER LIMITS AGAINST THE NET-POSITION LIMIT   "
          f"mean over {len(mx)} hours")
    print("=" * 92)
    print(out.to_string())
    print()
    print("  'overstatement' is the sum of a zone's per-border maxima divided")
    print("  by its maximum net position, in the same hour, from the same")
    print("  publisher. 1.00 would mean the borders are independent and the")
    print("  numbers can be added. They cannot.")
    print()

    # One hour written out in full, so the arithmetic can be checked by hand.
    z = args.zones[0] if args.zones[0] in CORE_BORDERS else "DE"
    t0 = mx["t"].iloc[0]
    cols = [f"border_{z}_{n}" for n in CORE_BORDERS[z]]
    print("=" * 92)
    print(f"WORKED EXAMPLE   {z}, {t0}")
    print("=" * 92)
    row = mx.set_index("t").loc[t0]
    for c in cols:
        print(f"  {c:<22}{row[c]:>10,.0f} MW")
    total = float(row[cols].sum())
    npmax = float(mn.loc[t0, f"max{z}"])
    print(f"  {'sum':<22}{total:>10,.0f} MW")
    print(f"  {'max net position':<22}{npmax:>10,.0f} MW")
    print(f"  {'overstatement':<22}{total / npmax:>10.2f} x")


if __name__ == "__main__":
    main()
