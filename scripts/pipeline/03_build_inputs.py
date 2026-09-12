"""Derive model inputs from the hourly frames built by step 02.

    python scripts/03_build_inputs.py

Writes into data/processed/:
    capacity_factors.parquet   p_max_pu for wind, solar, run-of-river
    cf_report.csv              mean CF and how often it needed clipping
    installed_capacity.csv     MW by zone, technology and year
    boundary_position.parquet  net export to the unmodelled world, per zone

Run 02 first - this reads its output.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402

from spread.config import PROCESSED, RAW, load_config        # noqa: E402
from spread.process import (                                 # noqa: E402
    boundary_net_position,
    capacity_factors,
    installed_capacity,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("inputs")


def main() -> None:
    cfg = load_config()
    zones = cfg["zones"]
    external = cfg["borders_external"]

    generation = pd.read_parquet(PROCESSED / "generation.parquet")
    net_flows = pd.read_parquet(PROCESSED / "net_flows.parquet")

    log.info("installed capacity register ...")
    cap = installed_capacity(RAW)
    cap.to_csv(PROCESSED / "installed_capacity.csv", index=False)
    log.info("  %d rows, years %s", len(cap), sorted(cap["year"].unique()))

    log.info("capacity factors ...")
    cf, report = capacity_factors(generation, cap)
    cf.to_parquet(PROCESSED / "capacity_factors.parquet")
    report.to_csv(PROCESSED / "cf_report.csv", index=False)

    log.info("boundary net positions ...")
    pos = boundary_net_position(net_flows, zones, external)
    pos.to_parquet(PROCESSED / "boundary_position.parquet")

    # ---- report ----
    print("\n" + "=" * 78)
    print("CAPACITY FACTORS  (mean over the period; clipping shows a stale register)")
    print("=" * 78)
    print(report.sort_values("column").to_string(index=False))

    print("\n" + "=" * 78)
    print("INSTALLED CAPACITY, GW  (latest year)")
    print("=" * 78)
    latest = cap[cap["year"] == cap["year"].max()]
    pivot = (latest.pivot_table(index="tech", columns="zone", values="mw")
             .div(1000).round(1).fillna(0))
    print(pivot.to_string())

    print("\n" + "=" * 78)
    print("BOUNDARY NET POSITION, MW   (+ve = zone exports to the unmodelled world)")
    print("=" * 78)
    summary = pd.DataFrame({
        "mean": pos.mean().round(0),
        "sd": pos.std().round(0),
        "p5": pos.quantile(.05).round(0),
        "p95": pos.quantile(.95).round(0),
        "pct_exporting": (100 * (pos > 0).mean()).round(1),
    })
    print(summary.to_string())
    print("\nA zone with no external borders correctly shows all zeros.")


if __name__ == "__main__":
    main()
