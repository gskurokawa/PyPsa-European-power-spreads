"""Base cases for the twelve sample days, all three models, then compare.

    python run_sample_days.py --dry-run
    python run_sample_days.py
    python run_sample_days.py --compare-only

The sample is the second Wednesday of each month of 2025, screened for data
completeness by screen_days.py and written to sample_days.csv.

WHY ONE RUN PER DAY RATHER THAN ONE RUN OF TWELVE
-------------------------------------------------
10_build_network.py takes a contiguous window (--start, --days), not a list of
dates. It does not need to take a list: this model has no coupling between
hours, so solving each day on its own gives exactly the prices those hours would
have had inside a combined solve. Twelve one-day runs cost twelve lots of
start-up overhead and nothing else, and the per-day outputs are more useful than
a single pooled one because they show how much the answer moves between days.

WHAT THE COMPARISON IS FOR
--------------------------
Not to check that twelve days reproduce the year - they will not. It is to check
that the GAP BETWEEN THE THREE MODELS survives the sample. If the NTC model sits
well below the CNEC model on these days as it does across the year, the sample
carries the effect and a Monte Carlo built on it is worth the compute. If the
three collapse together, the sample has averaged away what the study measures.

Separation uses the same eps = 0.5 EUR/MWh as validate.price_separation().
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "data" / "processed" / "runs"
BUILD = "scripts/pipeline/10_build_network.py"

ARMS = {"ntc": [], "mnp": ["--max-net-pos"], "cnec": ["--flow-based"]}
FULL_YEAR = {"ntc": "ntc-2025-base", "mnp": "mnp-2025-base",
             "cnec": "cnec-2025-base"}
BORDER = ("DE_LU", "FR")
EPS = 0.5


def sample_days(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"{path} not found - run screen_days.py first")
    return [str(d) for d in pd.read_csv(path)["date"]]


def tag_for(arm: str, day: str) -> str:
    return f"s12-{arm}-{day.replace('-', '')}"


def spread_stats(price: pd.DataFrame) -> tuple[float, float, int]:
    """Mean DE-FR spread, share of hours separated, hours counted."""
    a, b = BORDER
    if a not in price.columns or b not in price.columns:
        return float("nan"), float("nan"), 0
    diff = (price[a] - price[b]).dropna()
    if diff.empty:
        return float("nan"), float("nan"), 0
    return float(diff.mean()), float((diff.abs() > EPS).mean() * 100), len(diff)


def read_prices(tag: str):
    path = RUNS / tag / "prices.parquet"
    return pd.read_parquet(path) if path.exists() else None


def observed(days: list[str] | None):
    path = ROOT / "data" / "processed" / "prices.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame[frame.index.year == 2025]
    if days is not None:
        keep = frame.index.normalize().isin(pd.to_datetime(days, utc=True))
        frame = frame[keep]
    return frame


def compare(days: list[str]) -> None:
    rows = []

    obs_12 = observed(days)
    obs_yr = observed(None)
    for label, frame in (("observed, 12 days", obs_12),
                         ("observed, full year", obs_yr)):
        if frame is not None:
            mean, sep, n = spread_stats(frame)
            rows.append((label, mean, sep, n))

    for arm in ARMS:
        parts = [p for p in (read_prices(tag_for(arm, d)) for d in days)
                 if p is not None]
        if parts:
            pooled = pd.concat(parts).sort_index()
            mean, sep, n = spread_stats(pooled)
            rows.append((f"{arm}, 12 days", mean, sep, n))
        else:
            rows.append((f"{arm}, 12 days", float("nan"), float("nan"), 0))

        full = read_prices(FULL_YEAR[arm])
        if full is not None:
            mean, sep, n = spread_stats(full)
            rows.append((f"{arm}, full year", mean, sep, n))

    print(f"\nDE-FR, separation at |spread| > {EPS:g} EUR/MWh\n")
    print(f"  {'':<22} {'mean':>9}  {'separated':>10}  {'hours':>7}")
    for label, mean, sep, n in rows:
        m = "     -   " if pd.isna(mean) else f"{mean:9.2f}"
        s = "     -    " if pd.isna(sep) else f"{sep:9.1f}%"
        print(f"  {label:<22} {m}  {s}  {n:7d}")

    print("\nRead the gaps between the three models, not the levels. If the "
          "\nordering and rough spacing of ntc / mnp / cnec hold on twelve days"
          "\nas they do across the year, the sample carries the effect.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-file", default="sample_days.csv")
    ap.add_argument("--arm", choices=list(ARMS) + ["all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--bid-ladder", type=float, default=None,
                    help="passed to every cell, as run_shocks.py does")
    a = ap.parse_args()

    days = sample_days(ROOT / a.days_file)
    arms = ARMS if a.arm == "all" else {a.arm: ARMS[a.arm]}

    if a.compare_only:
        compare(days)
        return 0

    jobs = []
    for arm, extra in arms.items():
        for day in days:
            tag = tag_for(arm, day)
            if (RUNS / tag).exists():
                continue
            cmd = [sys.executable, BUILD, "--start", day, "--days", "1",
                   "--tag", tag] + extra
            if a.bid_ladder is not None:
                cmd += ["--bid-ladder", str(a.bid_ladder)]
            jobs.append((tag, cmd))

    done = len(days) * len(arms) - len(jobs)
    print(f"\n{len(days)} days x {len(arms)} models = "
          f"{len(days) * len(arms)} cells, {done} already on disk, "
          f"{len(jobs)} to run")

    if a.dry_run:
        for tag, cmd in jobs:
            print("   ", " ".join(cmd[1:]))
        return 0

    t0 = time.time()
    for i, (tag, cmd) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {tag}", flush=True)
        t1 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        print(f"  -> exit {r.returncode} in {(time.time()-t1)/60:.1f} min "
              f"| elapsed {(time.time()-t0)/60:.1f} min", flush=True)
        if r.returncode != 0:
            print("  FAILED - stopping so the cause is visible")
            return 1

    print(f"\ndone in {(time.time()-t0)/60:.1f} minutes")
    compare(days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
