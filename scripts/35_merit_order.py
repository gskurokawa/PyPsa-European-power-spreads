"""What does the supply stack look like, in EUR/MWh, zone by zone?

    python scripts/35_merit_order.py

Writes logs/merit_order.txt.  No solve - it builds the same inputs script 10
builds (about a second) and prints the stack they imply.

WHY
---
Script 34 found two things in the 2024 run that a summary statistic hides.

First, the modelled price distribution has a HOLE.  In every zone the count of
hours at or below zero, below 5, and below 20 EUR/MWh is exactly the same
number - 1,473 in Germany, 1,368 in France.  Not similar: identical.  So the
model never produces a price between 0 and 20, while reality produces hundreds
of such hours.  That is not a calibration issue, it is a missing rung on the
ladder: nothing in the fleet has a running cost in that band, so the price
jumps from the negative-bidding renewables straight to whatever comes next.

Second, France is PINNED.  Its modelled deciles 3 to 7 read 27.2, 27.3, 27.3,
27.4, 27.4 - five deciles, half the year, at one price - where the observed
series climbs 25.4, 40.2, 53.0, 63.9, 74.5.  Half of all French hours are set
by a single flat step in the merit order.

Both are properties of the COST DATA, not of the optimisation, so both are
visible here without solving anything.  This prints the ladder so the gap and
the step can be seen directly, and named.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "build10", ROOT / "scripts" / "10_build_network.py")
build10 = importlib.util.module_from_spec(spec)
sys.modules["build10"] = build10
spec.loader.exec_module(build10)

snapshots = pd.date_range("2024-01-01", periods=24 * 14, freq="h", tz="UTC")
inputs = build10.load_inputs(snapshots)

fleet = inputs["fleet"]
mc = inputs["marginal_costs"]
log = open(ROOT / "logs" / "merit_order.txt", "w", encoding="utf-8")


def out(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    log.write(line + "\n")


out(f"fleet: {len(fleet)} blocks, columns: {list(fleet.columns)}")
zcol = next((c for c in fleet.columns if c.lower() in
             ("zone", "bidding_zone", "bus")), None)
pcol = next((c for c in fleet.columns if "p_nom" in c.lower()
             or c.lower() in ("capacity", "capacity_mw", "mw")), None)
tcol = next((c for c in fleet.columns if c.lower() in
             ("carrier", "tech", "technology", "fuel_tech")), None)
out(f"using zone={zcol}  capacity={pcol}  tech={tcol}\n")

srmc_mean = mc.mean()          # average over the sample fortnight
fleet = fleet.copy()
fleet["srmc"] = fleet["gen_name"].map(srmc_mean)

BANDS = [(-100, 0), (0, 5), (5, 10), (10, 20), (20, 30), (30, 40),
         (40, 50), (50, 60), (60, 70), (70, 85), (85, 100),
         (100, 150), (150, 1e6)]

out("=" * 96)
out("1  THERMAL CAPACITY BY RUNNING-COST BAND, MW   (the ladder's rungs)")
out("=" * 96)
zones = sorted(fleet[zcol].dropna().unique()) if zcol else []
hdr = f"{'band EUR/MWh':<16}" + "".join(f"{z:>9}" for z in zones)
out(hdr)
for lo, hi in BANDS:
    sel = fleet[(fleet["srmc"] >= lo) & (fleet["srmc"] < hi)]
    row = [sel[sel[zcol] == z][pcol].sum() if pcol else 0 for z in zones]
    if sum(row) < 1:
        continue
    out(f"{f'{lo:>5.0f} - {hi:<6.0f}':<16}"
        + "".join(f"{v:>9,.0f}" for v in row))
out("\n  an empty band is a rung the price can never land on")

out("\n" + "=" * 96)
out("2  THE BIGGEST SINGLE STEPS  (one block, one price, lots of MW)")
out("=" * 96)
for z in zones:
    sub = fleet[fleet[zcol] == z].dropna(subset=["srmc"])
    if sub.empty:
        continue
    big = sub.nlargest(4, pcol)
    out(f"\n  {z}")
    for r in big.itertuples():
        out(f"    {getattr(r, pcol):>8,.0f} MW at "
            f"{getattr(r, 'srmc'):>7.1f} EUR/MWh   "
            f"{getattr(r, tcol) if tcol else ''}")

out("\n" + "=" * 96)
out("3  THE GAP: WHAT SITS BETWEEN 0 AND 20 EUR/MWh?")
out("=" * 96)
gap = fleet[(fleet["srmc"] >= 0) & (fleet["srmc"] < 20)]
if gap.empty:
    out("  NOTHING - the thermal fleet has no block in that band at all.")
else:
    out(f"  {len(gap)} block(s), {gap[pcol].sum():,.0f} MW total:")
    for r in gap.itertuples():
        out(f"    {getattr(r, zcol)} {getattr(r, tcol) if tcol else '':<28}"
            f"{getattr(r, pcol):>8,.0f} MW at {getattr(r, 'srmc'):>6.1f}")
out("\n  Renewables, hydro and the unmodelled generators are NOT in this")
out("  table - they are added separately in network.build - so the band may")
out("  be filled by them.  What matters is whether anything DISPATCHABLE")
out("  sits there: if not, the price steps straight over it.")

out("\n" + "=" * 96)
out("4  FRANCE IN DETAIL  (the 27.3 plateau)")
out("=" * 96)
fr = fleet[fleet[zcol].astype(str).str.startswith("FR")].dropna(subset=["srmc"])
fr = fr.sort_values("srmc")
cum = 0.0
for r in fr.itertuples():
    cum += getattr(r, pcol)
    out(f"    {getattr(r, 'srmc'):>7.1f}  {getattr(r, pcol):>8,.0f} MW   "
        f"cumulative {cum:>9,.0f}   {getattr(r, tcol) if tcol else ''}")
out(f"\n  French load averages about 49,000 MW, peaks near 80,000.")
out("  Where that lands on the ladder above is what sets the price, and a")
out("  single step wide enough to swallow half the year's load levels is")
out("  what produces five deciles at one number.")

log.close()
print("\nwritten to logs/merit_order.txt")
