"""Pull raw ENTSO-E data into data/raw/ as monthly parquet chunks.

Restartable: anything already cached is skipped, so if this dies partway
through - rate limit, dropped wifi, closed laptop - just run it again.

    python scripts/01_pull_entsoe.py --test          # one request, checks the token
    python scripts/01_pull_entsoe.py                 # everything
    python scripts/01_pull_entsoe.py --datasets load prices
    python scripts/01_pull_entsoe.py --force         # re-pull even if cached
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402
from entsoe import EntsoePandasClient                        # noqa: E402

from spread.config import RAW, api_key, load_config          # noqa: E402
from spread.entsoe_pull import (                             # noqa: E402
    ANNUAL_DATASETS,
    BORDER_DATASETS,
    ZONE_DATASETS,
    pull,
)

log = logging.getLogger("pull")


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "logs" / "pull.log", encoding="utf-8"),
        ],
    )


def smoke_test(client) -> None:
    """One small request. Confirms the token works before a long pull."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-03", tz="UTC")
    series = client.query_day_ahead_prices("DE_LU", start=start, end=end)
    print(f"\nGot {len(series)} German day-ahead prices for 1-2 Jan 2024.")
    print(f"  mean  EUR {series.mean():.2f}/MWh")
    print(f"  range EUR {series.min():.2f} to {series.max():.2f}/MWh")
    print("\nToken works. Run without --test to pull everything.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset: load prices generation outages flows scheduled ntc "
                         "capacity capacity_per_unit")
    ap.add_argument("--force", action="store_true", help="re-pull cached chunks")
    ap.add_argument("--test", action="store_true", help="single request to check the token")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config()
    # timeout is essential: entsoe-py defaults it to None, which means requests
    # waits forever if ENTSO-E accepts the connection then never answers.
    # retry_count is kept low because pull() has its own backoff around this.
    client = EntsoePandasClient(
        api_key=api_key(), timeout=30, retry_count=1, retry_delay=5,
    )

    if args.test:
        smoke_test(client)
        return

    start, end = cfg["period"]["start"], cfg["period"]["end"]
    zones = cfg["zones"]
    borders = cfg["borders_internal"] + cfg["borders_external"]
    wanted = set(args.datasets) if args.datasets else None

    # Physical flows are published per direction as non-negative series, so a
    # net flow needs BOTH directions: net(A->B) = flow(A->B) - flow(B->A).
    # NTC is directional too but we only use it as a capacity bound, so the
    # forward direction is enough there.
    reversed_borders = [">".join(reversed(b.split(">"))) for b in borders]

    jobs = []                                    # (dataset, key, annual)
    for name in ZONE_DATASETS:
        if wanted is None or name in wanted:
            jobs += [(name, z, False) for z in zones]
    for name in BORDER_DATASETS:
        if wanted is None or name in wanted:
            # Scheduled commercial exchange is only needed on the borders that
            # become Links. External borders enter as a fixed net position, so
            # pulling 19 more borders in both directions would double the job
            # count for nothing.
            keys = cfg["borders_internal"] if name == "scheduled" else borders
            jobs += [(name, b, False) for b in keys]
            if name in ("flows", "scheduled"):
                jobs += [(name, ">".join(reversed(b.split(">"))), False)
                         for b in keys]
    for name in ANNUAL_DATASETS:
        if wanted is None or name in wanted:
            jobs += [(name, z, True) for z in zones]

    log.info("Pulling %s to %s | %d jobs", start, end, len(jobs))
    t0 = time.time()
    rows = []

    for i, (dataset, key, annual) in enumerate(jobs, 1):
        log.info("[%d/%d] %s : %s", i, len(jobs), dataset, key)
        stats = pull(client, dataset, key, start, end, RAW,
                     annual=annual, force=args.force)
        log.info("        written %(written)d  skipped %(skipped)d  "
                 "empty %(empty)d  failed %(failed)d", stats)
        rows.append({"dataset": dataset, "key": key, **stats})

    summary = pd.DataFrame(rows)
    by_dataset = summary.groupby("dataset")[
        ["written", "skipped", "empty", "failed", "rows"]
    ].sum()

    print("\n" + "=" * 68)
    print(by_dataset.to_string())
    print("=" * 68)
    print(f"finished in {(time.time() - t0) / 60:.1f} min")

    failed = summary[summary["failed"] > 0]
    if not failed.empty:
        print("\nJobs with failures - re-run to retry just these:")
        print(failed[["dataset", "key", "failed"]].to_string(index=False))
        print("\n(Some external borders genuinely publish nothing. See logs/pull.log.)")


if __name__ == "__main__":
    main()
