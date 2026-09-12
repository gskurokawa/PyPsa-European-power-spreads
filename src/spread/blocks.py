"""Collapse the unit-level fleet into dispatch blocks.

Why this exists
---------------
A full model year came in at 674 s and 10.4 GB. At 16 GB that is exactly ONE
concurrent worker, so a 2,500-draw Monte Carlo is roughly 480 hours. Package E
does not run without this.

The LP's size is driven by generators x snapshots: 956 x 8,760 is 8.4 million
dispatch variables before the links, the reserve slacks and the net-position
constraints. Collapsing the fleet to roughly 150 blocks is about a sixfold cut.

Why it is not a compromise
--------------------------
Within one zone and technology, twelve hard-coal units whose efficiencies sit
within a percentage point of each other are already indistinguishable in the
dual - the price is set by the band, not by the unit. What aggregation destroys
is the ability to take one named unit out, and that matters for exactly one
thing: driver 4, thermal forced outages.

So outages stay at unit level. The draw is made against the real 865 units and
then applied to the block's available capacity, which preserves the mechanism -
a 900 MW unit tripping is a 900 MW step, not a smooth derate - while the LP
only ever sees the blocks. That split is the whole design.

Banding
-------
Blocks are formed within (zone, technology) by EFFICIENCY QUANTILE, not by a
fixed efficiency grid. Quantiles keep the number of blocks predictable and put
the boundaries where the fleet actually separates; a fixed grid would give
Poland fourteen hard-coal blocks and Austria none.

Capacity sums. Efficiency, emission factor and VOM are capacity-weighted, so
each block's SRMC is the capacity-weighted SRMC of what it replaces - which is
the quantity that sets the clearing price when the block is marginal.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WEIGHTED = ["efficiency", "co2_t_per_mwh_th", "vom_eur_per_mwh", "fixed_fuel_cost"]


def aggregate_fleet(fleet: pd.DataFrame, bands: int = 3) -> pd.DataFrame:
    """Return a block-level fleet with the same columns as the unit-level one.

    `bands` is the number of efficiency bands per (zone, technology). Three is
    the default because it keeps the merit order's shape - a cheap, a middle and
    an expensive band - at roughly a sixth of the generator count.
    """
    if fleet.empty:
        return fleet

    work = fleet.copy()
    work["capacity_mw"] = work["capacity_mw"].astype(float)

    def _band(group: pd.DataFrame) -> pd.Series:
        if len(group) <= bands:
            return pd.Series(range(len(group)), index=group.index)
        # rank rather than raw value: ties and clustered efficiencies would
        # otherwise collapse qcut into fewer bands than asked for
        ranks = group["efficiency"].rank(method="first")
        return pd.Series(
            pd.qcut(ranks, bands, labels=False, duplicates="drop"),
            index=group.index,
        )

    work["band"] = (work.groupby(["zone", "tech"], group_keys=False)
                        .apply(_band).astype(int))

    rows = []
    for (zone, tech, band), grp in work.groupby(["zone", "tech", "band"]):
        mw = float(grp["capacity_mw"].sum())
        if mw <= 0:
            continue
        w = grp["capacity_mw"].to_numpy(dtype=float)
        row = {
            "zone": zone,
            "tech": tech,
            "fuel": grp["fuel"].iloc[0],
            "name": f"{zone} {tech} band {band}",
            "capacity_mw": mw,
            "commissioned": float(np.average(
                pd.to_numeric(grp["commissioned"], errors="coerce")
                  .fillna(grp["commissioned"].median()).to_numpy(dtype=float),
                weights=w)),
            "units": len(grp),
        }
        for col in WEIGHTED:
            if col in grp:
                row[col] = float(np.average(grp[col].to_numpy(dtype=float), weights=w))
        rows.append(row)

    blocks = pd.DataFrame(rows)
    blocks["gen_name"] = (blocks["zone"] + " " + blocks["tech"] + " "
                          + blocks.groupby(["zone", "tech"]).cumcount().astype(str))
    log.info("fleet aggregated: %d units -> %d blocks (%d bands per zone/tech)",
             len(fleet), len(blocks), bands)
    return blocks


def compare(fleet: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """What the aggregation changed, per zone and technology.

    Capacity must be preserved exactly. Mean efficiency must be preserved to
    rounding. The spread of efficiency within a group is what is lost, and how
    much is lost is the number worth looking at before trusting any result.
    """
    def summarise(df: pd.DataFrame, label: str) -> pd.DataFrame:
        g = df.groupby(["zone", "tech"])
        w = lambda x: np.average(x["efficiency"], weights=x["capacity_mw"])  # noqa: E731
        out = pd.DataFrame({
            f"{label}_mw": g["capacity_mw"].sum().round(),
            f"{label}_n": g.size(),
            f"{label}_eff": g.apply(w).round(4),
            f"{label}_eff_sd": g["efficiency"].std().round(4),
        })
        return out

    return summarise(fleet, "unit").join(summarise(blocks, "block"), how="outer")
