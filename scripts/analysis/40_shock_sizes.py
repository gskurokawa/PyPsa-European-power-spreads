"""Record how big each driver shock ACTUALLY was, and where the size came from.

    python scripts/40_shock_sizes.py

Writes logs/shock_sizes.json for scripts/38_shock_analysis.py.

WHY THIS IS NOT A VARIANCE DECOMPOSITION
----------------------------------------
The intention was to size every shock at one standard deviation of its own
driver, so the responses could be compared and a variance decomposition
computed. That does not survive contact with the data, and the reason is worth
recording rather than hiding.

Gas and carbon are sized from the standard deviation of the DAILY series over
the 2024-2026 window: 23.9% and 10.6% of their means. Weather and French
nuclear were sized from the standard deviation of the ANNUAL mean, which with
two full years on record gives 1.4% and 0.6%. Those are not the same quantity.
The first pair measures volatility within the window; the second measures the
difference between two adjacent and unremarkable years, and for French nuclear
0.6% is noise, not uncertainty - the fleet produced about 279 TWh in 2022
during the stress-corrosion outages against roughly 360 TWh in 2024.

Two years cannot size year-to-year uncertainty for any driver. So the shares
across drivers are NOT reported as a variance decomposition. What is reported:

  * the RATIO of CNEC to NTC sensitivity, which is unaffected because both
    arms receive the identical shock and the size cancels. That is the
    section 13.1 test and the study's actual result.
  * the response at a STATED magnitude, with the magnitude and its provenance
    printed, so a reader can rescale it with their own view of how much a
    driver moves.

APPLIED SIZES
-------------
These are what the runs on disk actually used, not what a fresh derivation
would produce - the French nuclear cells were run at a hand-set 10% and have
not been re-run. Recording the applied value keeps the elasticity honest.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

applied, provenance = {}, {}

fp = pd.read_parquet(PROC / "fuel_prices.parquet")
for name, col in (("gas", "gas_eur_mwh_th"), ("co2", "co2_eur_t")):
    applied[name] = float(fp[col].std() / fp[col].mean())
    provenance[name] = ("1 sd of the daily series, 2024-2026 - volatility "
                        "within the window, not year-to-year uncertainty")

cf = pd.read_parquet(PROC / "capacity_factors.parquet")
ws = [c for c in cf.columns
      if any(k in str(c).lower() for k in ("wind", "solar"))]
yearly = cf[ws].groupby(cf.index.year).mean().mean(axis=1)
yearly = yearly[[y for y in yearly.index if (cf.index.year == y).sum() > 8000]]
applied["weather"] = (float(yearly.std() / yearly.mean())
                      if len(yearly) >= 2 else 0.05)
provenance["weather"] = (f"1 sd of the annual mean wind and solar capacity "
                         f"factor across {len(yearly)} full years - a "
                         f"two-point estimate, so a lower bound")

applied["FR nuclear"] = 0.10
provenance["FR nuclear"] = ("STATED ASSUMPTION, not derived. The runs on disk "
                            "used 10%. Deriving it from the 2024-2025 annual "
                            "means gives 0.6%, which is noise between two "
                            "similar years, not the driver's variability")

(ROOT / "logs").mkdir(exist_ok=True)
(ROOT / "logs" / "shock_sizes.json").write_text(
    json.dumps({"applied": applied, "provenance": provenance}, indent=2))

print("  shock sizes as applied, relative to each driver's own level:\n")
for k, v in applied.items():
    print(f"    {k:<13}{100*v:>7.1f} %   {provenance[k][:66]}")
print("\n  These are NOT on a common footing, so the cross-driver shares are")
print("  indicative only. The defensible result is the CNEC/NTC ratio, which")
print("  is independent of shock size. FINDINGS section 13.9 records this.")
print("\n  written to logs/shock_sizes.json")
