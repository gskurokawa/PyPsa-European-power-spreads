"""How much operating reserve, where, and does it hit more criteria?

    python scripts/19_tune_reserve.py --year 2024
    python scripts/19_tune_reserve.py --year 2024 --default 3 5 7 --cm 1.5 3

TWO parameters, swept as a grid. A multiplier on the reserve requirement in
zones with a capacity mechanism, and another for zones without. Not eight free
knobs - two numbers keyed on one published fact: Poland (2021), France (2017)
and Belgium (2021) run capacity markets, Germany, the Netherlands, Austria and
the Czech Republic do not. Switzerland is included on the other side of the same
argument: it sits outside EU market coupling and holds its own reserve.

Everything else is fixed.

Why this shape and not a free per-zone fit
------------------------------------------
A capacity market pays for firmness OUTSIDE the energy price, so scarcity rent
in the energy market is suppressed by design. A model with one system-wide
scarcity setting must therefore be wrong in opposite directions on the two
sides of that line. Two numbers test that claim; eight would fit it away, and
would spend the 2024 hold-out that makes the validation credible.

The prediction being tested, stated before the run: raising the no-mechanism
multiplier should lift the German level (-20.6%) and the German p10, and should
not move the Polish mean price (-14.2%, already passing) much because Poland
sits on the other side. If German level moves and nothing else does, scarcity
is a level problem only and the timing failures need their own diagnosis.

France is the honest test. France HAS a mechanism, so this rule pushes French
prices DOWN, and France is already the worst level failure at -30.2%. If France
gets worse here, that is evidence its problem is nuclear cost or the export
bound rather than scarcity - not a reason to exempt it.

Read the count, but read the columns too - a setting that passes more criteria
by breaking the two spread means is not progress.

RESULT (2024 hold-out, run 2026-09-05) - the hypothesis is REFUTED
------------------------------------------------------------------
Both predictions above were scored, and the interesting one failed.

1. FAILED. Raising the no-mechanism multiplier lifts the German LEVEL
   (-20.6% -> -14.1% at x7) but leaves the German p10 at EUR 0 vs EUR 10 at
   EVERY setting, unchanged to the euro. Reserve scarcity prices TIGHT hours,
   which live at the top of the distribution; the p10 failure is at the bottom
   and no amount of reserve can reach it. Level and the low tail are separate
   defects. This is the second independent sweep to say so.

2. CONFIRMED but negligible. France worsens as cm_x falls, by 0.5pp
   (-30.2% -> -30.7%). The whole cm/no-cm axis moves French level by half a
   point and Polish level by about one, against six points from the other axis.
   The capacity-mechanism split is real in sign and an order of magnitude too
   small to matter. Two parameters were not needed; this is one level knob.

And the level knob is a ZERO-SUM trade, not a gain. At x7 the German level
enters the +/-15% band - and the DE-FR mean spread leaves the +/-EUR 5 band
(+1.38 -> +5.29). 7/15 before, 7/15 after. There is a window near x6.5 where
both would sit just inside their bands, and taking it would be fitting a
parameter to one decimal place against the hold-out year to buy one criterion.
That is the thing this project exists not to do.

Kept at default 3.0 / cm 3.0 unless a reason arrives that is not the count.
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
log = logging.getLogger("reserve")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--bands", type=int, default=3)
    ap.add_argument("--chunk-days", type=int, default=30)
    ap.add_argument("--default", type=float, nargs="+",
                    default=[3.0, 5.0, 7.0],
                    help="multiplier for zones WITHOUT a capacity mechanism")
    ap.add_argument("--cm", type=float, nargs="+",
                    default=[1.5, 3.0],
                    help="multiplier for zones WITH a capacity mechanism")
    args = ap.parse_args()
    tee_output("reserve_sweep")

    zones = load_config()["zones"]
    snapshots = pd.date_range(f"{args.year}-01-01", f"{args.year + 1}-01-01",
                              freq="h", tz="UTC", inclusive="left")
    inputs = load_inputs(snapshots, bands=args.bands)
    actual = pd.read_parquet(PROCESSED / "prices.parquet").reindex(snapshots)

    raw = inputs.get("reserve_requirement_raw")
    if raw is None:
        raise SystemExit("operating_reserve is disabled in technology.yaml")
    cm_zones = set(inputs.get("capacity_market_zones", []))
    if not cm_zones:
        raise SystemExit("capacity_market_zones is empty - the grid would be "
                         "one parameter swept twice")
    log.info("capacity mechanism: %s", ", ".join(sorted(cm_zones)))
    log.info("no mechanism:       %s",
             ", ".join(z for z in zones if z not in cm_zones))

    n = build(inputs, snapshots, zones)
    bounds = inputs.get("net_position")
    np_fn = (net_position_limits(bounds, inputs.get('net_position_hourly'))
             if bounds is not None else None)
    model_sns = n.snapshots
    size = args.chunk_days * 24
    chunks = [model_sns[i:i + size] for i in range(0, len(model_sns), size)]

    settings = list(product(args.default, args.cm))
    log.info("%d settings x %d chunks", len(settings), len(chunks))

    rows = []
    for d_mult, c_mult in settings:
        per_zone = pd.Series({z: (c_mult if z in cm_zones else d_mult)
                              for z in zones})
        req = raw * per_zone

        extra = combine(np_fn,
                        reserve_constraint(req, inputs["reserve_tiers"],
                                           inputs["thermal_names"],
                                           inputs.get("reserve_hydro_credit")))
        t0 = time.time()
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

        rows.append({
            "no_cm_x": d_mult,
            "cm_x": c_mult,
            "req_DE_mw": round(float(req["DE_LU"].mean())),
            "req_FR_mw": round(float(req["FR"].mean())),
            "met": f"{met}/{len(checks)}",
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
        log.info("  no-cm x%.1f  cm x%.1f  ->  %s met   (%.0fs)",
                 d_mult, c_mult, rows[-1]["met"], time.time() - t0)

    print("\n" + "=" * 150)
    print("OPERATING RESERVE SWEEP   two parameters, fifteen criteria")
    print("  no_cm_x applies to " + ", ".join(z for z in zones if z not in cm_zones))
    print("  cm_x    applies to " + ", ".join(sorted(cm_zones)))
    print("=" * 150)
    frame = pd.DataFrame(rows).set_index(["no_cm_x", "cm_x"])
    print(frame.to_string())
    print("\n  Watch the two spread MEANS. They pass today; a setting that")
    print("  gains criteria elsewhere while breaking them has moved the model")
    print("  off the one thing it currently gets right.")
    print("  Watch FR_level too - the rule predicts it gets WORSE as cm_x falls.")

    out = ROOT / "logs" / "reserve_sweep.csv"
    out.parent.mkdir(exist_ok=True)
    frame.to_csv(out)
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
