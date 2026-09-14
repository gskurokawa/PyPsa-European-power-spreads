"""Measure the gas-carbon correlation properly, at several frequencies.

    python check_correlation.py

WHY THIS EXISTS
---------------
make_draws.py correlates the two draws using the correlation between the daily
gas and carbon PRICE LEVELS over 2024-2026, which came out at 0.316. That is the
wrong statistic for the job.

A draw is an addition to the year's price level. What it needs is the
correlation between two counterfactual level deviations: if gas had sat higher,
would carbon have. The correlation of two level series measures something else -
how the two series happened to move together over one particular window - and a
common drift in both, or opposite drifts, shows up there as correlation that
says nothing about the counterfactual.

A deviation in the annual level is the accumulation of that year's price
changes, and the correlation of sums of changes is the correlation of the
changes. So the correlation of CHANGES is what carries over, not the correlation
of levels.

The daily change correlation is near zero, but daily correlations between
commodity series are commonly attenuated by non-synchronous settlement and
short-term noise, so this reports weekly and monthly as well. Monthly is the
fair test at the horizon a draw represents; it is also the noisiest, having the
fewest observations, so the confidence interval is printed alongside.

WHAT TO DO WITH THE ANSWER
--------------------------
Take the monthly figure. If its interval comfortably contains zero and the point
estimate is small, set the correlation used in make_draws.py to 0. If monthly
comes back materially above the daily figure, the daily one was attenuated and
the monthly estimate is the one to use.

Either way the choice does not change the ratio between the three models'
dispersions, only the dispersions themselves - see section 8.4.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

GAS = "gas_eur_mwh_th"
CO2 = "co2_eur_t"


def fisher_ci(r: float, n: int, conf: float = 0.95):
    """95% interval for a correlation, via the Fisher z transform."""
    if n < 5 or abs(r) >= 1:
        return float("nan"), float("nan")
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    crit = 1.959963985 if conf == 0.95 else 1.644853627
    return math.tanh(z - crit * se), math.tanh(z + crit * se)


def report(label: str, x: pd.Series, y: pd.Series) -> None:
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)
    if n < 5:
        print(f"  {label:<26} only {n} observations - not reported")
        return
    r = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    lo, hi = fisher_ci(r, n)
    print(f"  {label:<26} {r:+.3f}   95% CI {lo:+.2f} to {hi:+.2f}   n = {n:,}")


def main() -> int:
    path = PROCESSED / "fuel_prices.parquet"
    if not path.exists():
        sys.exit("fuel_prices.parquet not found")

    fuel = pd.read_parquet(path)
    for col in (GAS, CO2):
        if col not in fuel.columns:
            sys.exit(f"column {col} not in fuel_prices.parquet")
    fuel.index = pd.to_datetime(fuel.index, errors="coerce")
    fuel = fuel[[GAS, CO2]].sort_index()

    print(f"\nfuel_prices.parquet  {fuel.index.min():%Y-%m-%d} to "
          f"{fuel.index.max():%Y-%m-%d},  {len(fuel):,} daily rows\n")

    print("correlation between the gas and carbon series")
    print("  (the level row is what make_draws.py currently uses)\n")

    report("LEVELS, daily", fuel[GAS], fuel[CO2])
    print()
    report("CHANGES, daily", fuel[GAS].diff(), fuel[CO2].diff())

    wk = fuel.resample("W").mean()
    report("CHANGES, weekly means", wk[GAS].diff(), wk[CO2].diff())

    mth = fuel.resample("ME").mean()
    report("CHANGES, monthly means", mth[GAS].diff(), mth[CO2].diff())

    qtr = fuel.resample("QE").mean()
    report("CHANGES, quarterly means", qtr[GAS].diff(), qtr[CO2].diff())

    yr = fuel.resample("YE").mean()
    print(f"\n  annual means available: {len(yr)} "
          f"({', '.join(str(d.year) for d in yr.index)})"
          f" - too few to correlate, which is the underlying problem")

    print("\nReading it: the monthly figure is the one to use. Its interval is"
          "\nwide because there are only about thirty observations; that width"
          "\nis the honest state of knowledge, not a defect of the measurement."
          "\nIf the interval contains zero, zero is a defensible choice and the"
          "\nsimpler one to justify.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
