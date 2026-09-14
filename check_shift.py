"""Verify --gas-shift/--co2-shift against the existing --shock path.

    python check_shift.py

Two things have to hold before a Monte Carlo built on the new flags means
anything:

    a zero shift changes nothing        chk-plain == chk-zero
    a one-sigma shift IS the old shock  chk-shock == chk-shift

The second is the one that matters. --shock gas+ adds one standard deviation of
the daily gas series; --gas-shift takes the number directly. Feed it that same
standard deviation and the two runs must agree to the digit, because they are
the same parallel shift arrived at by different routes. If they do not, the new
code path is doing something the old one does not, and every draw would inherit
it.

This script reports what it finds and prints the commands for whichever runs
are missing. It writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RUNS = PROCESSED / "runs"
DAY = "2025-01-08"
BUILD = "scripts/pipeline/10_build_network.py"
TOL = 1e-9


def prices(tag: str):
    path = RUNS / tag / "prices.parquet"
    return pd.read_parquet(path) if path.exists() else None


def compare(a: str, b: str, what: str) -> bool | None:
    left, right = prices(a), prices(b)
    missing = [t for t, f in ((a, left), (b, right)) if f is None]
    if missing:
        print(f"  {what}: not run yet ({', '.join(missing)})")
        return None

    if list(left.columns) != list(right.columns):
        print(f"  {what}: FAIL - different columns")
        return False
    if not left.index.equals(right.index):
        print(f"  {what}: FAIL - different snapshots")
        return False

    gap = (left - right).abs().to_numpy().max()
    if gap <= TOL:
        print(f"  {what}: identical")
        return True
    print(f"  {what}: FAIL - largest difference {gap:.6g} EUR/MWh")
    worst = (left - right).abs().max().sort_values(ascending=False)
    print(f"      worst zones: "
          f"{', '.join(f'{z} {v:.4g}' for z, v in worst.head(3).items())}")
    return False


def main() -> int:
    path = PROCESSED / "fuel_prices.parquet"
    if not path.exists():
        sys.exit("fuel_prices.parquet not found")
    fuel = pd.read_parquet(path)

    sd_gas = float(fuel["gas_eur_mwh_th"].std())
    sd_co2 = float(fuel["co2_eur_t"].std())
    idx = pd.to_datetime(fuel.index, errors="coerce")

    print(f"\nfuel_prices.parquet spans {idx.min():%Y-%m-%d} to "
          f"{idx.max():%Y-%m-%d}, {len(fuel):,} rows")
    print(f"  1 sd gas    {sd_gas:8.4f} EUR/MWh_th   "
          f"(mean {fuel['gas_eur_mwh_th'].mean():.2f})")
    print(f"  1 sd carbon {sd_co2:8.4f} EUR/t        "
          f"(mean {fuel['co2_eur_t'].mean():.2f})")

    print("\nchecks")
    zero_ok = compare("chk-plain", "chk-zero", "zero shift changes nothing")
    sigma_ok = compare("chk-shock", "chk-shift", "one sigma equals --shock gas+")

    if sigma_ok is None:
        print("\nrun these two, then this script again:\n")
        if prices("chk-shock") is None:
            print(f"  python {BUILD} --start {DAY} --days 1 "
                  f"--shock gas+ --tag chk-shock")
        if prices("chk-shift") is None:
            print(f"  python {BUILD} --start {DAY} --days 1 "
                  f"--gas-shift {sd_gas:.10f} --tag chk-shift")

    results = [r for r in (zero_ok, sigma_ok) if r is not None]
    if results and all(results) and sigma_ok:
        print("\nBoth checks pass. The drawn path and the pre-committed path "
              "agree,\nso a draw of n sigma is the --shock cell scaled by n.")
        return 0
    return 1 if any(r is False for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
