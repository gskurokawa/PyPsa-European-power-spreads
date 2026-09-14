"""Extract every border from the Monte Carlo runs already on disk.

    python mc_borders.py

run_mc.py reported one border, DE_LU-FR. The 300 runs it made each solved the
whole eight-zone model and saved every zone's hourly price, so the other twelve
borders cost nothing to add: this reads the same files again and writes one row
per border, model and draw.

No solving. Expect a couple of minutes for the file reading.

Output: mc_borders.csv, 13 borders x 3 models x 100 draws = 3,900 rows, plus a
row per border for the observed prices over the same twelve days.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RUNS = PROCESSED / "runs"

ARMS = ["ntc", "mnp", "cnec"]
EPS = 0.5                      # as in section 6.3

# The thirteen internal links of section 3, as zone pairs.
BORDERS = [
    ("DE_LU", "FR"), ("DE_LU", "PL"), ("DE_LU", "CZ"), ("DE_LU", "NL"),
    ("DE_LU", "BE"), ("DE_LU", "AT"), ("DE_LU", "CH"),
    ("FR", "BE"), ("FR", "CH"), ("NL", "BE"),
    ("AT", "CH"), ("AT", "CZ"), ("CZ", "PL"),
]


def stats(price: pd.DataFrame, a: str, b: str):
    if a not in price.columns or b not in price.columns:
        return None
    diff = (price[a] - price[b]).dropna()
    if diff.empty:
        return None
    return float(diff.mean()), float((diff.abs() > EPS).mean() * 100), len(diff)


def main() -> int:
    draws_path = ROOT / "draws.csv"
    if not draws_path.exists():
        sys.exit("draws.csv not found")
    draws = pd.read_csv(draws_path)

    rows = []

    # Observed prices over the same twelve days, for reference.
    days = pd.read_csv(ROOT / "sample_days.csv")["date"]
    obs_path = PROCESSED / "prices.parquet"
    if obs_path.exists():
        obs = pd.read_parquet(obs_path)
        obs.index = pd.to_datetime(obs.index, utc=True, errors="coerce")
        keep = obs.index.normalize().isin(pd.to_datetime(days, utc=True))
        obs = obs[keep]
        for a, b in BORDERS:
            s = stats(obs, a, b)
            if s:
                rows.append({"border": f"{a}-{b}", "arm": "observed", "draw": 0,
                             "gas_sigma": 0.0, "co2_sigma": 0.0,
                             "mean_spread": s[0], "separated_pct": s[1],
                             "hours": s[2]})

    missing = 0
    for arm in ARMS:
        for d in draws.itertuples():
            path = RUNS / f"mc-{arm}-{d.draw:04d}" / "prices.parquet"
            if not path.exists():
                missing += 1
                continue
            price = pd.read_parquet(path)
            for a, b in BORDERS:
                s = stats(price, a, b)
                if s:
                    rows.append({"border": f"{a}-{b}", "arm": arm,
                                 "draw": int(d.draw),
                                 "gas_sigma": float(d.gas_sigma),
                                 "co2_sigma": float(d.co2_sigma),
                                 "mean_spread": s[0], "separated_pct": s[1],
                                 "hours": s[2]})
        print(f"  {arm} done", flush=True)

    if missing:
        print(f"  {missing} run directories were absent and skipped")

    out = ROOT / "mc_borders.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False, float_format="%.6f")
    print(f"\n{len(frame):,} rows, {frame.border.nunique()} borders -> {out}")

    print("\nmean spread by border and model, EUR/MWh")
    piv = frame.pivot_table(index="border", columns="arm",
                            values="mean_spread", aggfunc="mean")
    print(piv.reindex(columns=["observed"] + ARMS).round(2))

    print("\nseparated hours by border and model, %")
    piv = frame.pivot_table(index="border", columns="arm",
                            values="separated_pct", aggfunc="mean")
    print(piv.reindex(columns=["observed"] + ARMS).round(1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
