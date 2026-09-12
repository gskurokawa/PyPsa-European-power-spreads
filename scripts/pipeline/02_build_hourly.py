"""Turn cached raw chunks into aligned hourly frames in data/processed/.

    python scripts/02_build_hourly.py

Writes:
    load.parquet            index=UTC hour, columns=zone
    prices.parquet          index=UTC hour, columns=zone
    generation.parquet      index=UTC hour, columns='ZONE|Technology'
    net_flows.parquet       index=UTC hour, columns=border, +ve = A exports to B
    link_capacity.csv       p_nom per border per direction, from flow percentiles
    coverage.csv            how complete every column actually is

Then prints a coverage summary. Read it before building a model on top:
the failure mode in this data is silence, not errors.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                              # noqa: E402

from spread.config import PROCESSED, RAW, load_config            # noqa: E402
from spread.process import (                                     # noqa: E402
    build_generation,
    blend_exchange,
    build_commercial_flows,
    build_net_flows,
    link_envelope,
    build_zone_series,
    coverage,
    hourly_index,
    link_capacity,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build")


def main() -> None:
    cfg = load_config()
    start, end = cfg["period"]["start"], cfg["period"]["end"]
    borders = cfg["borders_internal"] + cfg["borders_external"]
    PROCESSED.mkdir(parents=True, exist_ok=True)

    index = hourly_index(start, end)
    log.info("Target index: %d hourly steps, %s to %s\n",
             len(index), index[0].date(), index[-1].date())

    frames = {}

    log.info("load ...")
    frames["load"] = build_zone_series("load", "load", RAW, index)

    log.info("prices ...")
    frames["prices"] = build_zone_series("prices", "price", RAW, index)

    log.info("generation ...")
    frames["generation"] = build_generation(RAW, index)

    log.info("net flows ...")
    frames["net_flows"] = build_net_flows(RAW, index, borders)

    for name, frame in frames.items():
        path = PROCESSED / f"{name}.parquet"
        frame.to_parquet(path)
        log.info("  wrote %-16s %5d x %-4d  -> %s",
                 name, frame.shape[0], frame.shape[1], path.name)

    log.info("commercial exchange ...")
    frames["commercial_flows"] = build_commercial_flows(RAW, index, borders)

    # Links are sized from what the auction allocated, not from what crossed
    # the wire. See capacity_source in config/zones.yaml.
    source = cfg.get("capacity_source", "physical")
    outside = cfg.get("outside_market_coupling", [])
    # ONE canonical exchange series, written to disk, used by everything
    # downstream: the envelope, the static capacities, the net-position bounds,
    # the congestion diagnostic and the link calibration.
    #
    # Three scripts were choosing this independently and one of them - the
    # calibration - was still reading raw commercial exchange. It duly shrank
    # the Swiss borders to match a series that understates them, and the model
    # shed 87 GWh of Swiss load at VOLL. The same Switzerland failure had
    # already been fixed twice, in the net-position bounds and in the static
    # capacities, because the CHOICE was in three places instead of one.
    if source == "commercial" and not frames["commercial_flows"].empty:
        basis = blend_exchange(frames["commercial_flows"], frames["net_flows"], outside)
        log.info("  exchange basis: commercial, physical on borders touching %s",
                 ", ".join(outside) or "nothing")
    else:
        basis = frames["net_flows"]
        log.info("  exchange basis: physical flow")
    frames["exchange"] = basis
    for name in ("commercial_flows", "exchange"):
        frames[name].to_parquet(PROCESSED / f"{name}.parquet")
        log.info("  wrote %-16s %5d x %-4d", name,
                 frames[name].shape[0], frames[name].shape[1])

    env_fwd, env_rev = link_envelope(basis)
    env_fwd.to_parquet(PROCESSED / "link_env_fwd.parquet")
    env_rev.to_parquet(PROCESSED / "link_env_rev.parquet")
    log.info("  wrote %-16s hourly capacity, %d borders", "link_envelope",
             env_fwd.shape[1])

    caps = link_capacity(basis)
    caps.to_csv(PROCESSED / "link_capacity.csv")
    log.info("  wrote %-16s %5d borders", "link_capacity", len(caps))

    cov = pd.concat([coverage(f, n) for n, f in frames.items()], ignore_index=True)
    cov.to_csv(PROCESSED / "coverage.csv", index=False)

    # ---- summary ----
    print("\n" + "=" * 70)
    print("COVERAGE  (columns below 99% complete)")
    print("=" * 70)
    bad = cov[cov["pct_complete"] < 99].sort_values("pct_complete")
    if bad.empty:
        print("  every column at least 99% complete")
    else:
        print(bad[["dataset", "column", "missing", "pct_complete"]]
              .to_string(index=False))

    print("\n" + "=" * 70)
    print("LINK CAPACITY from observed flow (99th percentile, MW)")
    print("=" * 70)
    print(caps[["p_nom_fwd_mw", "p_nom_rev_mw", "max_fwd_mw", "max_rev_mw"]]
          .to_string())

    print("\n" + "=" * 70)
    print("SANITY: mean day-ahead price by zone and year, EUR/MWh")
    print("=" * 70)
    p = frames["prices"]
    print(p.groupby(p.index.year).mean().round(1).to_string())

    print("\n" + "=" * 70)
    print("SANITY: the spreads this project is about, EUR/MWh")
    print("=" * 70)
    for pair in [("DE_LU", "FR"), ("DE_LU", "PL")]:
        a, b = pair
        if a in p.columns and b in p.columns:
            s = (p[a] - p[b]).dropna()
            print(f"  {a}-{b:6s} mean {s.mean():7.2f}   sd {s.std():6.2f}   "
                  f"p5 {s.quantile(.05):7.2f}   p95 {s.quantile(.95):7.2f}   "
                  f"share of hours > 0: {100 * (s > 0).mean():.1f}%")


if __name__ == "__main__":
    main()
