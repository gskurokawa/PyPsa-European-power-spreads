"""Why is the modelled price distribution too flat? Answer per zone.

    python scripts/13_price_formation.py --days 30
    python scripts/13_price_formation.py --days 30 --zone DE_LU

Real prices in January-February 2025 spent 23-37% of hours above 150 EUR/MWh.
The model produced none. Real daily fuel prices changed almost nothing, so the
cause is the supply stack, not the fuel. This script separates the candidates:

1. PRICE DURATION CURVE, modelled against observed, by decile. Shows whether
   the model is missing the top, the bottom, or is compressed throughout.

2. MARGINAL TECHNOLOGY. Which carrier is partially loaded and therefore
   setting the price, as a share of hours. A zone that shows one technology in
   90% of hours has a stack the drivers cannot move it off.

3. HEADROOM IN THE TIGHTEST HOURS. Available-but-unused thermal capacity in the
   5% of hours with the highest price. This is the decisive number. If the
   model is carrying 10 GW spare at peak, no fuel price will ever produce a
   300 EUR hour, and the missing mechanism is one that withholds capacity:
   operating reserve, minimum up and down times, ramp rates - the things a unit
   commitment model has and an LP dispatch model does not.

4. UNREACHED STACK. Capacity whose SRMC sits above the highest price the model
   ever clears. Capacity the model owns but never uses.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from spread.config import PROCESSED, load_config             # noqa: E402
from spread.network import (build, combine, net_position_limits,  # noqa: E402
                            reserve_constraint)

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module                          # noqa: E402
load_inputs = import_module("10_build_network").load_inputs

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("pypsa", "linopy"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

DECILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
TIGHT = 0.05                      # share of hours treated as "tightest"


def dense(n, attr: str) -> pd.DataFrame:
    """p_max_pu / marginal_cost as a full snapshots x generators frame."""
    return n.get_switchable_as_dense("Generator", attr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--start", default="2025-01-13")
    ap.add_argument("--zone", default=None, help="restrict the detail tables")
    args = ap.parse_args()

    zones = load_config()["zones"]
    snapshots = pd.date_range(pd.Timestamp(args.start, tz="UTC"),
                              periods=args.days * 24, freq="h")
    inputs = load_inputs(snapshots)
    n = build(inputs, snapshots, zones)

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
    actual = pd.read_parquet(PROCESSED / "prices.parquet").reindex(snapshots)

    p = n.generators_t.p.copy()
    avail = dense(n, "p_max_pu").mul(n.generators.p_nom, axis=1)
    floor = dense(n, "p_min_pu").mul(n.generators.p_nom, axis=1)
    cost = dense(n, "marginal_cost")
    p.index = avail.index = floor.index = cost.index = snapshots

    bus = n.generators.bus
    carrier = n.generators.carrier.fillna("unknown")
    is_shed = n.generators.index.str.endswith(" shed")

    # The thermal fleet is the set of units from fleet.csv. Everything else on
    # a bus - wind, solar, run-of-river, hydro profiles, DSR, the boundary
    # import - is not dispatchable capacity and must not be counted as spare
    # or totalled as "thermal", which an earlier version of this script did.
    thermal = pd.Index(inputs["fleet"]["gen_name"]).intersection(n.generators.index)

    # ---- 1. price duration curve ----
    print("\n" + "=" * 78)
    print("PRICE DURATION CURVE, EUR/MWh   (quantile of the period)")
    print("=" * 78)
    rows = []
    for zone in zones:
        if zone not in price or zone not in actual:
            continue
        m, a = price[zone].dropna(), actual[zone].dropna()
        row = {"zone": zone}
        for q in DECILES:
            row[f"m{int(q * 100)}"] = round(float(m.quantile(q)))
            row[f"a{int(q * 100)}"] = round(float(a.quantile(q)))
        rows.append(row)
    dur = pd.DataFrame(rows).set_index("zone")
    print(dur[[c for c in dur.columns if c.startswith("m")]].to_string())
    print("\n  observed:")
    print(dur[[c for c in dur.columns if c.startswith("a")]].to_string())
    print("\n  m = modelled, a = actual. Compare column by column: a gap that")
    print("  widens toward the right is a missing top of stack, not a level error.")

    # ---- 2. marginal technology ----
    print("\n" + "=" * 78)
    print("MARGINAL TECHNOLOGY, share of hours   (the partially loaded unit)")
    print("=" * 78)
    tol = 1.0                                   # MW
    shares = {}
    for zone in zones:
        gens = n.generators.index[(bus == zone) & ~is_shed]
        if not len(gens):
            continue
        pz, az, cz, fz = p[gens], avail[gens], cost[gens], floor[gens]
        # Genuinely at the margin means free to move BOTH ways. A must-run unit
        # sitting on its floor is not setting the price, it is pinned there -
        # counting those made every zone look like it was on Steam Turbine in
        # 100% of hours, which was an artefact, not a finding.
        partial = (pz > fz + tol) & (pz < az - tol)
        # Among partially loaded units the dearest one is the price setter.
        # An hour with none is not an error: the price can be set by an import
        # or by the net-position bound, in which case no local unit is at the
        # margin - and how often that happens is itself a result.
        masked = cz.where(partial)
        has_setter = masked.notna().any(axis=1)
        idx = pd.Series("(no local unit - import or bound)",
                        index=masked.index, dtype=object)
        if has_setter.any():
            idx.loc[has_setter] = masked.loc[has_setter].idxmax(axis=1)
        tech = idx.map(lambda g: carrier[g] if g in carrier.index else g)
        shares[zone] = tech.value_counts(normalize=True).mul(100).round(1)
    marg = pd.DataFrame(shares).fillna(0.0)
    marg = marg.loc[marg.max(axis=1).sort_values(ascending=False).index]
    print(marg.head(12).to_string())

    # ---- 3. headroom in the tightest hours ----
    print("\n" + "=" * 78)
    print(f"HEADROOM IN THE TIGHTEST {TIGHT:.0%} OF HOURS, MW")
    print("=" * 78)
    load = inputs["load"].reindex(snapshots)
    rows = []
    for zone in zones:
        if zone not in price:
            continue
        tight = price[zone].nlargest(max(1, int(TIGHT * len(snapshots)))).index
        gens = thermal[bus[thermal] == zone]
        spare = (avail.loc[tight, gens] - p.loc[tight, gens]).clip(lower=0)
        spare_mw = float(spare.sum(axis=1).mean())
        peak_load = float(load[zone].loc[tight].mean()) if zone in load else np.nan
        rows.append({
            "zone": zone,
            "price_in_tight": round(float(price[zone].loc[tight].mean()), 1),
            "load_mw": round(peak_load),
            "spare_mw": round(spare_mw),
            "spare_%_of_load": round(100 * spare_mw / peak_load, 1) if peak_load else np.nan,
        })
    print(pd.DataFrame(rows).set_index("zone").to_string())

    print("\n  what the spare is made of, MW by carrier:")
    comp = {}
    for zone in zones:
        if zone not in price:
            continue
        tight = price[zone].nlargest(max(1, int(TIGHT * len(snapshots)))).index
        gens = thermal[bus[thermal] == zone]
        spare = (avail.loc[tight, gens] - p.loc[tight, gens]).clip(lower=0)
        comp[zone] = spare.mean().groupby(carrier[gens]).sum().round()
    print(pd.DataFrame(comp).fillna(0).astype(int).to_string())
    print("\n  A real system at 5% of hours from its peak is holding single-digit")
    print("  reserve. Double-digit spare here means the LP has capacity a real")
    print("  operator could not have called on - no notice, no minimum up time,")
    print("  no reserve obligation - and that is why the price never spikes.")

    # ---- 4. capacity the model never reaches ----
    print("\n" + "=" * 78)
    print("UNREACHED STACK: capacity priced above the highest cleared price")
    print("=" * 78)
    rows = []
    for zone in zones:
        if zone not in price:
            continue
        top = float(price[zone].max())
        gens = thermal[bus[thermal] == zone]
        mean_cost = cost[gens].mean()
        cap = n.generators.loc[gens, "p_nom"]
        above = cap[mean_cost > top].sum()
        above150 = cap[mean_cost > 150].sum()
        rows.append({
            "zone": zone,
            "max_price": round(top, 1),
            "cap_above_max_price_mw": round(float(above)),
            "cap_above_150_mw": round(float(above150)),
            "total_thermal_mw": round(float(cap.sum())),
        })
    print(pd.DataFrame(rows).set_index("zone").to_string())
    print("\n  Capacity sitting above the clearing price is not a bug - every")
    print("  system has peakers. It becomes one when the system is never tight")
    print("  enough to call them, because then the top of the distribution is")
    print("  unreachable by construction and no driver can produce a spike.")


if __name__ == "__main__":
    main()
