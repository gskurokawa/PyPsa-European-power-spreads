"""Sweep the operating reserve curve against the observed price distribution.

    python scripts/14_calibrate_reserve.py --days 30
    python scripts/14_calibrate_reserve.py --days 30 --loads 0.04 0.06 0.08 --scales 0.5 1.0

The reserve demand curve is the one calibrated, behavioural piece of this
model, so it is worth being explicit about how its numbers were chosen rather
than nudging them by hand until something looked right.

The network is built once and re-solved per parameter set, because building
costs 37 seconds and solving costs 29.

What is scored
--------------
The price DURATION CURVE, not the mean. A scarcity mechanism that lifts the
mean is trivial to build and worthless; the test is whether the shape matches
across quantiles, and in particular whether the median stays put while the
upper tail extends. Score is the mean absolute error across the 50th, 90th,
95th and 99th percentiles, averaged over zones, with the median weighted
double - drifting the middle of the distribution to buy a better tail is the
failure mode this is guarding against.

Honesty note
------------
This fits three numbers to 2025 prices, which are also what the model is
validated on. That is defensible only because the parameter count is tiny and
because the chosen curve must then reproduce 2024 UNCHANGED. Run
scripts/10_build_network.py --year 2024 before believing any of it.
"""
from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402

from spread.config import PROCESSED, load_config             # noqa: E402
from spread.network import (build, combine, net_position_limits,  # noqa: E402
                            reserve_constraint)

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module                          # noqa: E402
load_inputs = import_module("10_build_network").load_inputs

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("pypsa", "linopy", "spread.network"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
log = logging.getLogger("reserve")

QUANTILES = [0.50, 0.90, 0.95, 0.99]
WEIGHTS = {0.50: 2.0, 0.90: 1.0, 0.95: 1.0, 0.99: 1.0}


def score(price: pd.DataFrame, actual: pd.DataFrame, zones: list[str]) -> float:
    total, weight = 0.0, 0.0
    for zone in zones:
        if zone not in price or zone not in actual:
            continue
        m, a = price[zone].dropna(), actual[zone].dropna()
        for q in QUANTILES:
            w = WEIGHTS[q]
            total += w * abs(float(m.quantile(q)) - float(a.quantile(q)))
            weight += w
    return total / weight if weight else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--start", default="2025-01-13")
    ap.add_argument("--contingency", type=float, nargs="+",
                    default=[0.0, 2000.0, 4000.0, 6000.0],
                    help="MW held against loss of the largest unit. PyPSA-Eur "
                         "uses 4000; 0 removes the term entirely")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0],
                    help="multipliers on the tier prices")
    args = ap.parse_args()

    zones = load_config()["zones"]
    snapshots = pd.date_range(pd.Timestamp(args.start, tz="UTC"),
                              periods=args.days * 24, freq="h")
    inputs = load_inputs(snapshots)
    actual = pd.read_parquet(PROCESSED / "prices.parquet").reindex(snapshots)

    base_req = inputs["reserve_requirement"]
    if base_req is None:
        raise SystemExit("operating_reserve is disabled in technology.yaml")

    import yaml
    tech = yaml.safe_load(open(ROOT / "config" / "technology.yaml", encoding="utf-8"))
    res_cfg = tech["operating_reserve"]
    # requirement = epsilon_load*load + epsilon_vres*potential + contingency.
    # Strip the constant so the sweep can vary it on its own; the epsilons stay
    # at PyPSA-Eur's published values and are not swept, because they are the
    # part of this that is sourced rather than fitted.
    variable_part = base_req - float(res_cfg.get("contingency", 0.0))

    n = build(inputs, snapshots, zones)
    bounds = inputs.get("net_position")
    np_fn = (net_position_limits(bounds, inputs.get('net_position_hourly'))
             if bounds is not None else None)

    rows = []
    for cont in args.contingency:
        req = (variable_part + cont).clip(lower=0.0)
        for scale in args.scales:
            tiers = deepcopy(res_cfg["tiers"])
            for t in tiers:
                t["price"] = float(t["price"]) * scale

            n.optimize(
                solver_name="highs",
                solver_options={"threads": 1, "output_flag": False},
                extra_functionality=combine(
                    np_fn,
                    reserve_constraint(req, tiers, inputs["thermal_names"],
                                       inputs.get("reserve_hydro_credit")),
                ),
            )
            price = n.buses_t.marginal_price.copy()
            price.index = snapshots

            de = price["DE_LU"] - price["FR"]
            a_de = actual["DE_LU"] - actual["FR"]
            pl = price["DE_LU"] - price["PL"]
            a_pl = actual["DE_LU"] - actual["PL"]

            rows.append({
                "contingency": cont,
                "price_scale": scale,
                "score": round(score(price, actual, zones), 1),
                "DE_mean": round(float(price["DE_LU"].mean()), 1),
                "DE_p50": round(float(price["DE_LU"].quantile(0.50))),
                "DE_p95": round(float(price["DE_LU"].quantile(0.95))),
                "DE_p99": round(float(price["DE_LU"].quantile(0.99))),
                "PL_p50": round(float(price["PL"].quantile(0.50))),
                "PL_p95": round(float(price["PL"].quantile(0.95))),
                "FR_sd": round(float((de).std()), 1),
                "PL_sd": round(float((pl).std()), 1),
                "FR_pos%": round(100 * float((de > 0).mean()), 1),
            })
            log.info("  contingency %5.0f MW  price x%.2f  ->  score %.1f",
                     cont, scale, rows[-1]["score"])

    out = pd.DataFrame(rows).sort_values("score")
    print("\n" + "=" * 92)
    print("RESERVE CURVE SWEEP   (scored on the price duration curve, median weighted double)")
    print("=" * 92)
    print(out.to_string(index=False))

    print("\n  target:  DE_p50 %d   DE_p95 %d   DE_p99 %d   DE-FR sd %.1f (pos %.1f%%)"
          "   DE-PL sd %.1f" % (
              actual["DE_LU"].quantile(0.50), actual["DE_LU"].quantile(0.95),
              actual["DE_LU"].quantile(0.99),
              (actual["DE_LU"] - actual["FR"]).std(),
              100 * ((actual["DE_LU"] - actual["FR"]) > 0).mean(),
              (actual["DE_LU"] - actual["PL"]).std()))
    print("\n  Pick on the duration curve, not on the spread sd - fitting the sd")
    print("  directly is fitting the answer. Then set the winner in")
    print("  config/technology.yaml and validate on 2024 unchanged.")


if __name__ == "__main__":
    main()
