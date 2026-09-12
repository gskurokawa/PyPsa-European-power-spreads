"""Choose the nuclear bid-ladder spread on the CALIBRATION year.

    python scripts/41_tune_ladder.py --year 2024
    python scripts/41_tune_ladder.py --year 2024 --ladder 0 10 20 25 30 40

ONE parameter, fifteen criteria. `bid_ladder.Nuclear` is the EUR/MWh spread
across a zone's nuclear stack: units are ordered largest first and given an
adder running linearly from 0 to `spread` across cumulative capacity, so the
marginal unit's offer depends on how deep into the fleet demand reaches
(FINDINGS 6.6).

Why this script exists
----------------------
The ladder was originally chosen on 2025 while 2024 was held out. FINDINGS 4.1
now runs the split the other way - 2024 calibrates, 2025 is the out-of-sample
test - because several other decisions had already been taken after looking at
2024 (the `must_run` nuclear floor, the net-position source rule in script 12,
and the decision to add a ladder at all). Relabelling the years without moving
this parameter would leave the model fitted on what is now the test year, which
is the same defect wearing the opposite hat. So the ladder is re-chosen here,
on 2024, and 2025 is then run once with the answer.

What to watch, and what would be cheating
-----------------------------------------
The ladder has a physical justification - nuclear is energy-limited, so the
last megawatt of a fleet is not offered at the cost of the first - and a
defensible value is one where the French price DISTRIBUTION stops sitting on a
single number, not one that maximises the criteria count. Read FR_level and the
criteria count together, and prefer a round number in a flat region over the
argmax. Picking the argmax to one euro would be fitting a parameter to the
scoreboard, which is what the split exists to prevent.

Germany has no nuclear after April 2023, so no setting here can move the German
level. If it appears to, something else changed.
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402

from spread.config import PROCESSED, load_config, tee_output  # noqa: E402
from spread.network import (build, combine, net_position_limits,  # noqa: E402
                            reserve_constraint)
from spread.validate import stopping_rule                    # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module                          # noqa: E402
_m = import_module("10_build_network")
load_inputs = _m.load_inputs

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("pypsa", "linopy", "spread.network", "spread.process"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
log = logging.getLogger("ladder")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024,
                    help="the CALIBRATION year. Do not point this at 2025.")
    ap.add_argument("--bands", type=int, default=3,
                    help="efficiency bands per zone and technology. 0 means "
                         "the full unit-level fleet. This is NOT neutral for "
                         "this parameter: the ladder spreads a technology's "
                         "units across cumulative capacity, so banding the "
                         "fleet to 3 blocks turns a smooth ramp into 3 steps. "
                         "Sweep at 3 to find the region, confirm at 0.")
    ap.add_argument("--chunk-days", type=int, default=30)
    ap.add_argument("--ladder", type=float, nargs="+",
                    default=[0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
                    help="EUR/MWh spreads to try for Nuclear")
    args = ap.parse_args()
    tee_output("ladder_sweep")

    if args.year == 2025:
        log.warning("2025 is the TEST year in FINDINGS 4.1. Sweeping a "
                    "parameter on it spends the out-of-sample claim.")

    zones = load_config()["zones"]
    snapshots = pd.date_range(f"{args.year}-01-01", f"{args.year + 1}-01-01",
                              freq="h", tz="UTC", inclusive="left")
    actual = pd.read_parquet(PROCESSED / "prices.parquet").reindex(snapshots)

    log.info("%d ladder settings on %d snapshots of %d, %s fleet",
             len(args.ladder), len(snapshots), args.year,
             f"{args.bands}-band" if args.bands else "full unit-level")

    rows = []
    for spread_eur in args.ladder:
        t0 = time.time()

        # The ladder enters through marginal_costs(), which load_inputs()
        # computes once - so inputs must be rebuilt per setting, not reused.
        inputs = load_inputs(snapshots, bands=(args.bands or None),
                             bid_ladder=spread_eur)
        n = build(inputs, snapshots, zones)

        bounds = inputs.get("net_position")
        np_fn = (net_position_limits(bounds, inputs.get("net_position_hourly"))
                 if bounds is not None else None)
        req = inputs.get("reserve_requirement")
        res_fn = (reserve_constraint(req, inputs["reserve_tiers"],
                                     inputs["thermal_names"],
                                     inputs.get("reserve_hydro_credit"))
                  if req is not None and inputs.get("reserve_tiers") else None)
        extra = combine(np_fn, res_fn)

        model_sns = n.snapshots
        size = args.chunk_days * 24
        chunks = [model_sns[i:i + size] for i in range(0, len(model_sns), size)]
        for chunk in chunks:
            n.optimize(snapshots=chunk, solver_name="highs",
                       solver_options={"threads": 1, "output_flag": False},
                       extra_functionality=extra)
            try:
                n._model = None
            except Exception:                                 # noqa: BLE001
                pass
            gc.collect()

        price = n.buses_t.marginal_price.copy()
        price.index = snapshots
        checks = stopping_rule(price, actual)
        met = sum(1 for _, _, ok in checks if ok)

        def val(fragment):
            return next((v for nm, v, _ in checks if fragment in nm), "-")

        # The point of the ladder is that France stops pricing at one number.
        # A criteria count cannot see that, so measure it directly: how many
        # distinct euros the middle of the French distribution spans.
        fr = price["FR"].dropna() if "FR" in price else pd.Series(dtype=float)
        fr_d3_d7 = (round(float(fr.quantile(0.7) - fr.quantile(0.3)), 1)
                    if len(fr) else float("nan"))

        rows.append({
            "ladder_eur": spread_eur,
            "met": f"{met}/{len(checks)}",
            "FR_d3_d7_span": fr_d3_d7,
            "DE_level": val("DE_LU mean price"),
            "FR_level": val("FR mean price"),
            "PL_level": val("PL mean price"),
            "DE_p10": val("DE_LU p10"),
            "DE_p90": val("DE_LU p90"),
            "PL_p10": val("PL p10"),
            "DE-FR_mean": val("DE_LU-FR mean spread"),
            "DE-PL_mean": val("DE_LU-PL mean spread"),
            "DE-FR_sd": val("DE_LU-FR spread sd"),
            "DE-PL_sd": val("DE_LU-PL spread sd"),
            "DE-FR_corr": val("DE_LU-FR hourly"),
            "DE-PL_corr": val("DE_LU-PL hourly"),
        })
        log.info("  ladder %5.1f  ->  %s met   FR d3-d7 span %.1f   (%.0fs)",
                 spread_eur, rows[-1]["met"], fr_d3_d7, time.time() - t0)

        del n, inputs
        gc.collect()

    print("\n" + "=" * 150)
    print(f"NUCLEAR BID-LADDER SWEEP   one parameter, fifteen criteria, "
          f"calibration year {args.year}")
    print("=" * 150)
    frame = pd.DataFrame(rows).set_index("ladder_eur")
    print(frame.to_string())
    print("\n  FR_d3_d7_span is the euro distance between the 30th and 70th")
    print("  percentile of the French price. At ladder 0 it is near zero -")
    print("  that is the 63 GW plateau of FINDINGS 6.6. Observed 2024 is about")
    print("  34 EUR. This is the quantity the ladder exists to repair; the")
    print("  criteria count is a check on the side effects, not the target.")
    print("\n  Choose a round value in a flat region. Do NOT take the argmax.")

    suffix = f"_b{args.bands}" if args.bands else "_full"
    out = ROOT / "logs" / f"ladder_sweep{suffix}.csv"
    out.parent.mkdir(exist_ok=True)
    frame.to_csv(out)
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
