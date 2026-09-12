"""Does each zone's observed data balance? No model involved.

    python scripts/17_zone_balance.py
    python scripts/17_zone_balance.py --year 2025

Generation + net imports - load should be about zero in every hour, for every
zone, in the DATA. Nothing here is modelled: this reads what ENTSO-E published
and checks it against itself.

Why it exists
-------------
The Netherlands has been the largest single-zone error in the model for
several runs - net position inverted by roughly 3.4 GW against both the
commercial and the physical series - while Dutch generation validates fine
against ENTSO-E per technology (gas 1.11, hard coal 0.95). Those two facts
cannot both be dispatch problems. If the published data does not balance for
NL, the error is in an input and no amount of merit-order work will find it.

Expect a small negative residual everywhere: ENTSO-E reports pumped-storage
GENERATION in the generation tables but its CONSUMPTION separately, so the
pumping load is missing from this sum. A few hundred MW in a hydro zone is
normal. A few gigawatts is not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402

from spread.config import PROCESSED, load_config             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    zones = cfg["zones"]

    load = pd.read_parquet(PROCESSED / "load.parquet")
    gen = pd.read_parquet(PROCESSED / "generation.parquet")
    # Physical flow, not commercial: this is an energy balance, and only the
    # physical series is supposed to close.
    flows = pd.read_parquet(PROCESSED / "net_flows.parquet")

    if args.year:
        for f in (load, gen, flows):
            f.drop(f.index[f.index.year != args.year], inplace=True)

    rows = []
    for zone in zones:
        gcols = [c for c in gen.columns if c.startswith(f"{zone}|")]
        g = gen[gcols].sum(axis=1)

        exports = pd.Series(0.0, index=flows.index)
        internal, external = [], []
        for border in flows.columns:
            a, _, b = border.partition(">")
            if a == zone:
                exports = exports + flows[border].fillna(0.0)
                (internal if b in zones else external).append(border)
            elif b == zone:
                exports = exports - flows[border].fillna(0.0)
                (internal if a in zones else external).append(border)

        idx = g.index.intersection(load.index).intersection(flows.index)
        if zone not in load or len(idx) < 100:
            continue
        resid = (g.reindex(idx) - load[zone].reindex(idx) - exports.reindex(idx))
        rows.append({
            "zone": zone,
            "load_mw": round(float(load[zone].reindex(idx).mean())),
            "gen_mw": round(float(g.reindex(idx).mean())),
            "net_export_mw": round(float(exports.reindex(idx).mean())),
            "residual_mw": round(float(resid.mean())),
            "resid_%_load": round(100 * float(resid.mean())
                                  / float(load[zone].reindex(idx).mean()), 1),
            "resid_sd": round(float(resid.std())),
            "borders": len(internal) + len(external),
        })

    out = pd.DataFrame(rows).set_index("zone")
    print("\n" + "=" * 88)
    print("OBSERVED ENERGY BALANCE   generation + imports - load, MW")
    print("=" * 88)
    print(out.to_string())
    print("\n  residual = generation - load - net exports. Should be slightly")
    print("  NEGATIVE: pumped-storage consumption is published separately and")
    print("  is missing from the generation sum, so a hydro zone legitimately")
    print("  shows a few hundred MW. Anything past about 5% of load is an input")
    print("  error - a missing technology, a mis-signed border, or a load series")
    print("  that is not the same quantity as the generation series.")

    print("\n" + "=" * 88)
    print("TECHNOLOGIES PRESENT PER ZONE   (a missing one shows up as a residual)")
    print("=" * 88)
    techs = sorted({c.split("|", 1)[1] for c in gen.columns})
    grid = pd.DataFrame(
        {z: [f"{gen[f'{z}|{t}'].mean():,.0f}" if f"{z}|{t}" in gen else "-"
             for t in techs] for z in zones},
        index=techs,
    )
    print(grid.to_string())


if __name__ == "__main__":
    main()
