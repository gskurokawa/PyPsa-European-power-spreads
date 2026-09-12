"""Why does the weather shock inflate the spread sd sevenfold?

    python scripts/39_weather_diag.py

The shock swaps 2025's capacity factors onto 2024's load and fuel prices. It
moved the mean DE-FR spread by 16-22 EUR/MWh - larger than every other driver
- but took the standard deviation from 27.8 to 196.3 while barely moving the
p90. A mean driven by a handful of extreme hours rather than by a shifted
distribution is an artefact, not a driver.

Hypothesis: swapping a different year's renewable output onto this year's
demand decouples wind from the load it actually coincided with, so some hours
are far shorter than anything 2024 experienced. With no storage, and external
borders fixed to 2024's flows, the model has nowhere to go and sheds load at
VOLL. A few VOLL hours would produce exactly this signature.

This tests that directly rather than assuming it.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "processed" / "runs"
LOG = open(ROOT / "logs" / "weather_diag.txt", "w", encoding="utf-8")


def out(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    LOG.write(line + "\n")


base = pd.read_parquet(RUNS / "cnec-2024-base" / "prices.parquet")
wx = pd.read_parquet(RUNS / "cnec-2024-weather" / "prices.parquet")

out("=" * 88)
out("1  HOW EXTREME DOES THE WEATHER RUN GET?")
out("=" * 88)
out(f"  {'zone':<8}{'run':<10}{'max':>10}{'>500':>7}{'>1000':>7}"
    f"{'>3000':>7}{'min':>10}{'<-100':>7}")
for z in base.columns:
    for label, df in (("base", base), ("weather", wx)):
        v = df[z].dropna()
        out(f"  {z:<8}{label:<10}{v.max():>10.0f}{int((v > 500).sum()):>7}"
            f"{int((v > 1000).sum()):>7}{int((v > 3000).sum()):>7}"
            f"{v.min():>10.0f}{int((v < -100).sum()):>7}")

out("\n" + "=" * 88)
out("2  WHAT HAPPENS TO THE DE-FR SPREAD IF THE EXTREMES ARE EXCLUDED?")
out("=" * 88)
for label, df in (("base", base), ("weather", wx)):
    s = (df["DE_LU"] - df["FR"]).dropna()
    out(f"\n  {label}")
    out(f"    all hours        mean {s.mean():>8.2f}   sd {s.std():>8.2f}")
    for cap in (500, 1000, 3000):
        k = s[(df["DE_LU"].abs() < cap) & (df["FR"].abs() < cap)]
        out(f"    both |p| < {cap:<5} mean {k.mean():>8.2f}   sd {k.std():>8.2f}"
            f"   ({len(s)-len(k)} hours dropped)")
    out(f"    median           {s.median():>8.2f}   "
        f"IQR {s.quantile(.75)-s.quantile(.25):>8.2f}")

out("\n  If the mean survives the exclusion but the sd collapses, the shock is")
out("  real and only its dispersion is contaminated. If the mean collapses")
out("  too, the whole result was those few hours.")

out("\n" + "=" * 88)
out("3  IS IT LOAD SHEDDING?")
out("=" * 88)
g = RUNS / "cnec-2024-weather" / "generation.parquet"
gb = RUNS / "cnec-2024-base" / "generation.parquet"
if g.exists() and gb.exists():
    for label, f in (("base", gb), ("weather", g)):
        d = pd.read_parquet(f)
        shed = [c for c in d.columns if "shed" in str(c).lower()]
        tot = float(d[shed].sum().sum()) if shed else 0.0
        hrs = int((d[shed].sum(axis=1) > 0.1).sum()) if shed else 0
        out(f"  {label:<9} load shed {tot:>12,.0f} MWh over {hrs:>5} hours"
            f"   ({len(shed)} shedding generators)")

out("\n" + "=" * 88)
out("4  HOW DIFFERENT ARE THE TWO WEATHER YEARS?")
out("=" * 88)
cf = pd.read_parquet(ROOT / "data" / "processed" / "capacity_factors.parquet")
a = cf[cf.index.year == 2024]
b = cf[cf.index.year == 2025]
out(f"  capacity_factors columns: {len(cf.columns)}")
out(f"  {'column':<34}{'2024 mean':>11}{'2025 mean':>11}{'ratio':>8}")
for c in list(cf.columns)[:16]:
    m1, m2 = float(a[c].mean()), float(b[c].mean())
    out(f"  {str(c):<34}{m1:>11.3f}{m2:>11.3f}"
        f"{(m2/m1 if m1 > 1e-9 else float('nan')):>8.2f}")
out("\n  A ratio near 1.0 means the years differ in TIMING, not level - which")
out("  is what decouples renewable output from the load it coincided with.")

LOG.close()
print("\nwritten to logs/weather_diag.txt")
