"""Where in the distribution does the missing EUR 16/MWh live?

    python scripts/34_where_is_the_gap.py --tag 2024-base

Reads the SAVED run - no re-solve, about a second.  Scripts 13 and 15 answer
related questions but rebuild and re-solve the model to do it, which is why
this exists now that runs persist.

Writes logs/gap_<tag>.txt as well as printing.

WHAT SCRIPT 33 ALREADY RULED OUT
--------------------------------
The fuel and carbon series are correct for 2024 - gas 25.9 to 45.2, CO2 62.5
to 70.6, fully populated - so the cost stack is not too low.  And the top 1%
of observed German hours contributes only EUR 3.5 of the EUR 78.5 mean, so
missing scarcity cannot carry a EUR 16 shortfall on its own.

It also found the fact that reframes this.  In 2024 the observed German price
sat EUR 3.6 BELOW the model's own CCGT running cost; in 2025 it sat EUR 1.4
above.  So 2024 was a year in which gas was frequently NOT marginal.  A model
carrying too much cheap supply is therefore too low by more in 2024 than in
2025 - which is the pattern in the results.

The decile table below settles where the gap actually is:
  concentrated in the top decile  -> missing scarcity, fix the reserve ladder
  spread evenly across all ten    -> a constant level error
  concentrated in the bottom      -> too much cheap supply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="2024-base")
args = ap.parse_args()

RUN = PROC / "runs" / args.tag
if not RUN.exists():
    raise SystemExit(f"no run at {RUN}")

log = open(ROOT / "logs" / f"gap_{args.tag}.txt", "w", encoding="utf-8")


def out(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    log.write(line + "\n")


mod = pd.read_parquet(RUN / "prices.parquet")
obs = pd.read_parquet(PROC / "prices.parquet").reindex(mod.index)
zones = [z for z in mod.columns if z in obs.columns]

out("=" * 96)
out("1  MODELLED MINUS OBSERVED, BY DECILE OF EACH SORTED SERIES (EUR/MWh)")
out("=" * 96)
out(f"{'zone':<7}" + "".join(f"{'d'+str(i):>8}" for i in range(1, 11))
    + f"{'mean':>9}")
for z in zones:
    m = np.sort(mod[z].dropna().values)
    a = np.sort(obs[z].dropna().values)
    n = min(len(m), len(a)); m, a = m[:n], a[:n]
    cuts = np.array_split(np.arange(n), 10)
    out(f"{z:<7}" + "".join(f"{m[c].mean()-a[c].mean():>8.1f}" for c in cuts)
        + f"{m.mean()-a.mean():>9.1f}")

out("")
out("share of each zone's TOTAL shortfall contributed by each decile:")
out(f"{'zone':<7}" + "".join(f"{'d'+str(i):>8}" for i in range(1, 11)))
for z in zones:
    m = np.sort(mod[z].dropna().values)
    a = np.sort(obs[z].dropna().values)
    n = min(len(m), len(a)); m, a = m[:n], a[:n]
    cuts = np.array_split(np.arange(n), 10)
    g = np.array([(m[c] - a[c]).sum() for c in cuts]); t = g.sum()
    out(f"{z:<7}" + "".join(f"{100*x/t:>7.0f}%" for x in g))

out("")
out("=" * 96)
out("2  THE LEVELS THEMSELVES, DECILE BY DECILE")
out("=" * 96)
for z in ("DE_LU", "FR", "PL", "CZ"):
    if z not in zones:
        continue
    m = np.sort(mod[z].dropna().values)
    a = np.sort(obs[z].dropna().values)
    n = min(len(m), len(a)); m, a = m[:n], a[:n]
    cuts = np.array_split(np.arange(n), 10)
    out(f"\n{z}")
    out("  decile  " + "".join(f"{'d'+str(i):>8}" for i in range(1, 11)))
    out("  model   " + "".join(f"{m[c].mean():>8.1f}" for c in cuts))
    out("  actual  " + "".join(f"{a[c].mean():>8.1f}" for c in cuts))

out("")
out("=" * 96)
out("3  CHEAP HOURS: HOW MANY, AND HOW CHEAP")
out("=" * 96)
out(f"{'zone':<7}{'mod<=0':>8}{'obs<=0':>8}{'mod<5':>8}{'obs<5':>8}"
    f"{'mod<20':>8}{'obs<20':>8}{'mod<40':>8}{'obs<40':>8}")
for z in zones:
    m, a = mod[z].dropna(), obs[z].dropna()
    out(f"{z:<7}" + "".join(f"{int(v):>8}" for v in
                            [(m <= 0).sum(), (a <= 0).sum(),
                             (m < 5).sum(), (a < 5).sum(),
                             (m < 20).sum(), (a < 20).sum(),
                             (m < 40).sum(), (a < 40).sum()]))

out("")
out("=" * 96)
out("4  GENERATION MIX, MODELLED, ANNUAL TWh BY ZONE AND CARRIER")
out("=" * 96)
gp = RUN / "generation.parquet"
if gp.exists():
    g = pd.read_parquet(gp)
    tw = (g.sum() / 1e6).sort_values(ascending=False)
    for k, v in tw.items():
        if v > 0.5:
            out(f"  {str(k):<40}{v:>8.1f}")
    out(f"\n  total modelled generation: {tw.sum():.1f} TWh")

try:
    ld = pd.read_parquet(PROC / "load.parquet").reindex(mod.index)
    out(f"  total load in the same hours: {ld.sum().sum()/1e6:.1f} TWh")
    out("  per zone, TWh: " + ", ".join(f"{c} {ld[c].sum()/1e6:.0f}"
                                        for c in ld.columns))
except Exception as exc:                                      # noqa: BLE001
    out(f"  load unreadable: {str(exc)[:60]}")

log.close()
print(f"\nwritten to logs/gap_{args.tag}.txt")
