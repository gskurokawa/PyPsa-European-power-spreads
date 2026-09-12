"""Validate dispatch by technology against observed generation.

    python scripts/15_dispatch_validation.py --days 30
    python scripts/15_dispatch_validation.py --days 30 --zone DE_LU

Everything so far has been validated on prices. Prices are a derived quantity -
the dual of a constraint - so a model can get them roughly right for the wrong
reasons, and can get them wrong without telling you which technology is
misbehaving. Generation is the primary observable and it is per technology, so
it says where the error lives.

The immediate question this was written for: German prices at the 5th
percentile come out at 96 EUR/MWh against an actual 72. Seventy-two is BELOW
German lignite's short-run cost of about 87 - lignite ran at a loss rather
than shut down. Whether the model reproduces that is a must-run question, and
the way to answer it is to look at lignite output in the cheapest hours rather
than to nudge the must_run share until the price moves.

Three tables:
  1. Mean output by technology, modelled against observed, with correlation.
  2. Output in the CHEAPEST 10% of hours - where must-run behaviour shows.
  3. Output in the TIGHTEST 10% of hours - where availability shows.

A technology that matches on the mean but not in the cheap hours has a
must-run problem. One that matches in cheap hours but falls short in tight
hours has an availability problem. The two are separable and this separates
them.
"""
from __future__ import annotations

import argparse
import logging
import sys
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

# Fleet carrier -> the ENTSO-E production type it should be compared against.
# The three gas technologies share one observed series, so they are summed
# before comparison; ENTSO-E does not split gas by turbine type.
CARRIER_TO_ENTSOE = {
    "Lignite":       "Fossil Brown coal/Lignite",
    "Hard Coal":     "Fossil Hard coal",
    "CCGT":          "Fossil Gas",
    "OCGT":          "Fossil Gas",
    "Steam Turbine": "Fossil Gas",
    "Oil":           "Fossil Oil",
    "Nuclear":       "Nuclear",
    "Bioenergy":     "Biomass",
    "Waste":         "Waste",
}
SHARE = 0.10


def table(title: str, modelled: pd.DataFrame, observed: pd.DataFrame,
          zones: list[str], hours: dict | None = None) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)
    rows = []
    for zone in zones:
        # CCGT, OCGT and Steam Turbine all map to "Fossil Gas" - ENTSO-E does
        # not split gas by turbine type - so deduplicate on the target series
        # or every gas row prints three times.
        for entsoe in sorted(set(CARRIER_TO_ENTSOE.values())):
            mcol, ocol = (zone, entsoe), f"{zone}|{entsoe}"
            if mcol not in modelled or ocol not in observed:
                continue
            idx = hours[zone] if hours else modelled.index
            m = modelled[mcol].reindex(idx).dropna()
            o = observed[ocol].reindex(idx).dropna()
            common = m.index.intersection(o.index)
            if len(common) < 10:
                continue
            m, o = m.loc[common], o.loc[common]
            rows.append({
                "zone": zone, "tech": entsoe,
                "mod_mw": round(float(m.mean())),
                "obs_mw": round(float(o.mean())),
                "diff_mw": round(float(m.mean() - o.mean())),
                "ratio": round(float(m.mean() / o.mean()), 2) if o.mean() else float("nan"),
                "corr": round(float(m.corr(o)), 2),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print("  nothing to compare")
        return
    print(df.set_index(["zone", "tech"]).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--start", default="2025-01-13")
    ap.add_argument("--zone", default=None)
    args = ap.parse_args()

    zones = load_config()["zones"]
    if args.zone:
        zones = [args.zone]
    all_zones = load_config()["zones"]

    snapshots = pd.date_range(pd.Timestamp(args.start, tz="UTC"),
                              periods=args.days * 24, freq="h")
    inputs = load_inputs(snapshots)
    n = build(inputs, snapshots, all_zones)

    bounds = inputs.get("net_position")
    req = inputs.get("reserve_requirement")
    n.optimize(
        solver_name="highs",
        solver_options={"threads": 1, "output_flag": False},
        extra_functionality=combine(
            net_position_limits(bounds, inputs.get('net_position_hourly'))
            if bounds is not None else None,
            reserve_constraint(req, inputs["reserve_tiers"],
                               inputs["thermal_names"],
                               inputs.get("reserve_hydro_credit"))
            if req is not None and inputs.get("reserve_tiers") else None,
        ),
    )

    price = n.buses_t.marginal_price.copy()
    price.index = snapshots
    p = n.generators_t.p.copy()
    p.index = snapshots

    # modelled output aggregated to (zone, ENTSO-E production type)
    carrier = n.generators.carrier
    bus = n.generators.bus
    cols = {}
    for gen in n.generators.index:
        entsoe = CARRIER_TO_ENTSOE.get(carrier.get(gen))
        if entsoe is None:
            continue
        cols.setdefault((bus[gen], entsoe), []).append(gen)
    modelled = pd.DataFrame({k: p[v].sum(axis=1) for k, v in cols.items()})
    modelled.columns = pd.MultiIndex.from_tuples(modelled.columns)

    observed = pd.read_parquet(PROCESSED / "generation.parquet").reindex(snapshots)

    table("MEAN OUTPUT BY TECHNOLOGY, MW   (whole period)",
          modelled, observed, zones)

    k = max(1, int(SHARE * len(snapshots)))
    cheap = {z: price[z].nsmallest(k).index for z in zones if z in price}
    tight = {z: price[z].nlargest(k).index for z in zones if z in price}

    table(f"CHEAPEST {SHARE:.0%} OF HOURS   (must-run shows here)",
          modelled, observed, zones, cheap)
    table(f"TIGHTEST {SHARE:.0%} OF HOURS   (availability shows here)",
          modelled, observed, zones, tight)

    print("\n  ratio < 1 in the cheap hours = the model switches plant off that")
    print("  reality kept running at a loss: raise must_run for that technology.")
    print("  ratio < 1 in the tight hours  = the model cannot call plant that")
    print("  reality did call: the availability cap is too tight.")
    print("  A low corr with a ratio near 1 means the level is right and the")
    print("  timing is wrong, which is a merit-order problem, not a volume one.")


if __name__ == "__main__":
    main()
