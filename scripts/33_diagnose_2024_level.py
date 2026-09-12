"""Why is every 2024 price 14-30% low, when 2025 is near-exact?

    python scripts/33_diagnose_2024_level.py

The shortfall is uniform across all eight zones - EUR 14 to 21 per MWh
everywhere - which rules out a fleet or availability error in any one zone
and points at something they share.  Two candidates, and this separates them:

  A  THE COST STACK IS TOO LOW.  If the fuel and carbon series feeding
     marginal cost are wrong for 2024, every hour is priced too cheaply and
     the whole distribution shifts down.  Testable without running the model:
     compare the SRMC of a reference gas plant, month by month, against the
     observed German price.  In a gas-set market those track closely.

  B  THE TOP OF THE STACK IS MISSING.  Poland shows 136 high-price hours
     modelled against 716 observed, Czechia 122 against 534.  Scarcity hours
     sit far above any plant's fuel cost, so missing them drags the MEAN down
     without touching the cost stack at all.  Testable by asking how much of
     the gap lives in the top few per cent of hours.

The answer decides the fix.  A is a data repair; B is the reserve ladder in
section 6.1 being calibrated on a year that was less tight than 2024.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

PROC = ROOT / "data" / "processed"

# a reference CCGT and hard-coal unit, for turning fuel into EUR/MWh
CCGT_EFF, CCGT_EF = 0.58, 0.20      # efficiency, tCO2 per MWh of fuel
COAL_EFF, COAL_EF = 0.40, 0.34


def show(name, d, n=14):
    print(f"\n  {name}: shape {d.shape}")
    if isinstance(d.index, pd.DatetimeIndex):
        print(f"    {d.index.min()} .. {d.index.max()}  tz={d.index.tz}")
    print(f"    columns: {', '.join(map(str, d.columns[:n]))}"
          f"{' ...' if len(d.columns) > n else ''}")


print("=" * 88)
print("1  WHAT THE FUEL AND CARBON SERIES ACTUALLY CONTAIN")
print("=" * 88)
fp = pd.read_parquet(PROC / "fuel_prices.parquet")
show("fuel_prices.parquet", fp)

num = fp.select_dtypes("number")
yearly = num.groupby(num.index.year).mean().round(2)
print("\n  annual means:")
print(yearly.to_string())

print("\n  monthly means, 2024 vs 2025:")
for c in num.columns[:8]:
    m = num[c].groupby([num.index.year, num.index.month]).mean()
    y24 = [m.get((2024, k), np.nan) for k in range(1, 13)]
    y25 = [m.get((2025, k), np.nan) for k in range(1, 13)]
    print(f"    {c:<16} 2024  " + " ".join(f"{v:6.1f}" if v == v else "   n/a"
                                            for v in y24))
    print(f"    {'':<16} 2025  " + " ".join(f"{v:6.1f}" if v == v else "   n/a"
                                            for v in y25))

print("\n  coverage - a partly-empty 2024 that was filled forward would")
print("  look fine in an annual mean and be wrong in the months that matter:")
for c in num.columns[:8]:
    for y in (2024, 2025):
        s = num.loc[num.index.year == y, c]
        if len(s):
            print(f"    {c:<16} {y}  {100*s.notna().mean():5.1f} % non-null, "
                  f"{s.nunique():>5} distinct values, "
                  f"{'CONSTANT' if s.nunique() <= 1 else ''}")


print("\n" + "=" * 88)
print("2  DOES THE COST STACK EXPLAIN THE OBSERVED PRICE?  (candidate A)")
print("=" * 88)
gas = next((c for c in num.columns if "gas" in c.lower()
            or "ttf" in c.lower()), None)
co2 = next((c for c in num.columns if "co2" in c.lower()
            or "eua" in c.lower() or "carbon" in c.lower()), None)
coal = next((c for c in num.columns if "coal" in c.lower()
             or "api" in c.lower()), None)
print(f"  using gas={gas}  co2={co2}  coal={coal}")

prices = pd.read_parquet(PROC / "prices.parquet")
show("prices.parquet", prices)
de = next((c for c in prices.columns if str(c).startswith("DE")), None)

if gas and co2 and de:
    srmc_g = num[gas] / CCGT_EFF + num[co2] * CCGT_EF / CCGT_EFF
    srmc_c = (num[coal] / COAL_EFF + num[co2] * COAL_EF / COAL_EFF
              if coal else None)
    df = pd.DataFrame({"obs_DE": prices[de], "ccgt": srmc_g})
    if srmc_c is not None:
        df["coal"] = srmc_c
    df = df.dropna(subset=["obs_DE"])
    mm = df.groupby([df.index.year, df.index.month]).mean().round(1)
    print("\n  monthly: observed German price vs the model's own cost stack")
    print("  (in a gas-set market these track; a persistent gap IS the bug)")
    print(mm.loc[2024].to_string() if 2024 in mm.index.get_level_values(0)
          else "  no 2024")
    print("\n  2025 for comparison:")
    print(mm.loc[2025].to_string() if 2025 in mm.index.get_level_values(0)
          else "  no 2025")
    for y in (2024, 2025):
        if y in mm.index.get_level_values(0):
            g = (mm.loc[y]["obs_DE"] - mm.loc[y]["ccgt"]).mean()
            print(f"\n  {y}: observed DE price minus CCGT SRMC = "
                  f"{g:+.1f} EUR/MWh on average")
    print("  If that gap is much larger in 2024 than 2025, the fuel/carbon")
    print("  series are too low for 2024 and candidate A is confirmed.")


print("\n" + "=" * 88)
print("3  HOW MUCH OF THE GAP LIVES IN THE EXPENSIVE HOURS?  (candidate B)")
print("=" * 88)
print("  Observed 2024 prices only - no model run needed.  If a large share")
print("  of the annual mean comes from the top few per cent of hours, then a")
print("  model that cannot produce those hours is low by roughly that much,")
print("  whatever its cost stack does.\n")
obs = prices[prices.index.year == 2024]
for z in [c for c in obs.columns][:8]:
    s = obs[z].dropna()
    if not len(s):
        continue
    total = s.mean()
    parts = []
    for q in (0.99, 0.95, 0.90):
        thr = s.quantile(q)
        top = s[s >= thr]
        parts.append(f"top {100*(1-q):>2.0f}%: "
                     f"{top.sum()/len(s):>5.1f} ({100*top.sum()/s.sum():>4.1f}%)")
    print(f"    {str(z):<8} mean {total:>6.1f}   " + "   ".join(parts))
print("\n  read as: contribution to the annual mean, EUR/MWh, and share of it")
