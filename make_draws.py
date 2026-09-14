"""Draw correlated gas and carbon shifts for the Monte Carlo.

    python make_draws.py                 # 100 correlated draws
    python make_draws.py --n 200
    python make_draws.py --independent   # for variance attribution instead

WHAT IS DRAWN
-------------
A pair of numbers per draw: how many EUR to add to the whole daily gas series,
and how many to add to the carbon series. Those go straight to
10_build_network.py's --gas-shift and --co2-shift, which apply the same parallel
shift that --shock applies at exactly one standard deviation. A draw of n sigma
is therefore the corresponding --shock cell scaled by n, which is what
check_shift.py verifies.

THE DISTRIBUTION, AND WHY ADDITIVE AND NORMAL
---------------------------------------------
Prices are bounded below and have a fatter upper tail than lower, which argues
for a multiplicative log-normal draw. Two things make the simpler additive
normal the better choice here.

The first is consistency: --shock already adds a fixed number of EUR, and every
figure in sections 7 and 8 is expressed in those units. An additive draw keeps a
draw and a published shock cell on one scale.

The second is that the objection does not bite at this dispersion. One standard
deviation of gas is about 24% of its mean, so a draw would have to fall below
roughly four sigma to push the price negative, which will not happen in a few
hundred draws. The script checks and reports it rather than assuming.

A study interested in the far tail - a crisis scenario, or a 1-in-100 year -
would need the log-normal. This one is asking what the spread does under
ordinary variation, and over plus or minus two sigma the two distributions are
almost indistinguishable.

CORRELATION
-----------
Gas and carbon move together: expensive gas pushes generation toward coal, coal
emits more, and allowance demand rises. Drawing them independently would
generate combinations that do not occur.

The correlation is measured on WEEKLY CHANGES, not on price levels. An earlier
version used the correlation between the two daily price LEVELS, which came out
at 0.316, and that was the wrong statistic. A draw is an addition to the year's
price level, so what it needs is the correlation between two counterfactual
level deviations. A deviation in the annual level is the accumulation of that
year's price changes, and the correlation of sums of changes is the correlation
of the changes; the correlation of two level series measures something else,
how those two series happened to move together over one window, and a common
drift in both appears there as correlation that says nothing about the
counterfactual.

Weekly rather than daily because the daily figure is attenuated: on this data
daily changes correlate at 0.023 and weekly at 0.222, the gap being settlement
noise in each series swamping a real relationship. Weekly rather than monthly
because monthly has about thirty observations and an interval running from
-0.21 to +0.49, wide enough to contain both the weekly estimate and zero. If
changes are roughly independent from one week to the next, the correlation of
annual sums equals the correlation of weekly changes, so weekly estimates the
same quantity as monthly and estimates it better.

check_correlation.py reports all five frequencies with their intervals.

The cost is that a regression on correlated draws cannot cleanly separate how
much of the spread's variation came from gas and how much from carbon. If that
attribution is wanted, --independent produces a second draw file for it. The two
answer different questions and both are cheap.

SAMPLING
--------
Latin hypercube rather than plain random: with only two dimensions, stratifying
each one spreads a small number of draws far more evenly than independent
sampling, which clumps. The correlation is imposed afterwards by a Cholesky
factor, which slightly degrades the stratification of the second dimension; that
is the usual trade and it still beats plain random at this sample size.

Uses numpy and the standard library only.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

GAS = "gas_eur_mwh_th"
CO2 = "co2_eur_t"

_NORMAL = statistics.NormalDist()


def latin_hypercube(n: int, dims: int, rng) -> np.ndarray:
    """Stratified uniforms in [0,1): one point per bin, shuffled per dimension."""
    cuts = (np.arange(n)[:, None] + rng.random((n, dims))) / n
    for d in range(dims):
        rng.shuffle(cuts[:, d])
    return cuts


def to_normal(u: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF, elementwise, without scipy."""
    flat = [_NORMAL.inv_cdf(float(x)) for x in u.ravel()]
    return np.asarray(flat).reshape(u.shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--independent", action="store_true",
                    help="set the correlation to zero, for attribution")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    path = PROCESSED / "fuel_prices.parquet"
    if not path.exists():
        sys.exit("fuel_prices.parquet not found")
    fuel = pd.read_parquet(path)
    idx = pd.to_datetime(fuel.index, errors="coerce")

    sd_gas = float(fuel[GAS].std())
    sd_co2 = float(fuel[CO2].std())
    mean_gas = float(fuel[GAS].mean())
    mean_co2 = float(fuel[CO2].mean())

    weekly = fuel[[GAS, CO2]].resample("W").mean().diff().dropna()
    rho_weekly = float(weekly[GAS].corr(weekly[CO2]))
    rho_level = float(fuel[GAS].corr(fuel[CO2]))
    rho_change = float(fuel[GAS].diff().corr(fuel[CO2].diff()))
    rho = 0.0 if a.independent else rho_weekly

    print(f"\nfuel_prices.parquet  {idx.min():%Y-%m-%d} to {idx.max():%Y-%m-%d}"
          f",  {len(fuel):,} rows")
    print(f"  gas     mean {mean_gas:7.2f}   1 sd {sd_gas:7.4f}  "
          f"({sd_gas / mean_gas:5.1%} of mean)")
    print(f"  carbon  mean {mean_co2:7.2f}   1 sd {sd_co2:7.4f}  "
          f"({sd_co2 / mean_co2:5.1%} of mean)")
    print(f"\n  correlation of weekly changes {rho_weekly:+.3f}   <- used "
          f"({len(weekly)} observations)")
    print(f"  correlation of daily changes  {rho_change:+.3f}   "
          f"attenuated by settlement noise")
    print(f"  correlation of price levels   {rho_level:+.3f}   "
          f"not the right statistic - see the docstring")
    if a.independent:
        print("  drawing INDEPENDENTLY (correlation forced to 0)")

    rng = np.random.default_rng(a.seed)
    z = to_normal(latin_hypercube(a.n, 2, rng))

    # Impose the correlation, then scale each margin by its own standard
    # deviation. A unit-variance pair times sd is a draw in EUR.
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = z @ chol.T
    gas_shift = z[:, 0] * sd_gas
    co2_shift = z[:, 1] * sd_co2

    draws = pd.DataFrame({
        "draw": np.arange(1, a.n + 1),
        "gas_shift": gas_shift,
        "co2_shift": co2_shift,
        "gas_sigma": z[:, 0],
        "co2_sigma": z[:, 1],
    })

    out = ROOT / (a.out or ("draws_independent.csv" if a.independent
                            else "draws.csv"))
    draws.to_csv(out, index=False, float_format="%.6f")

    achieved = float(np.corrcoef(gas_shift, co2_shift)[0, 1])
    print(f"\n{a.n} draws, seed {a.seed}")
    print(f"  gas    {gas_shift.min():+7.2f} to {gas_shift.max():+7.2f} EUR "
          f"({z[:, 0].min():+.2f} to {z[:, 0].max():+.2f} sd)")
    print(f"  carbon {co2_shift.min():+7.2f} to {co2_shift.max():+7.2f} EUR "
          f"({z[:, 1].min():+.2f} to {z[:, 1].max():+.2f} sd)")
    print(f"  correlation achieved in the sample {achieved:+.3f} "
          f"(target {rho:+.3f})")

    # A draw that would make a fuel free is not a shock, it is a different
    # model. Report it rather than let clip(lower=0) hide it.
    floor_gas = fuel[GAS].min() + gas_shift.min()
    floor_co2 = fuel[CO2].min() + co2_shift.min()
    if floor_gas < 0 or floor_co2 < 0:
        print(f"\n  WARNING: the most negative draw takes a daily value below "
              f"zero\n           (gas {floor_gas:.2f}, carbon {floor_co2:.2f})."
              f" clip(lower=0) in\n           load_inputs would silently "
              f"truncate it. Reduce --n or\n           reconsider the "
              f"distribution before running.")
    else:
        print(f"  lowest daily value after the most negative draw: "
              f"gas {floor_gas:.2f}, carbon {floor_co2:.2f}  (no clipping)")

    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
