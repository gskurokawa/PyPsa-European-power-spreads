"""Monte Carlo over the twelve sample days: run the draws, then read them.

    python run_mc.py --dry-run
    python run_mc.py                              # draws.csv, all three models
    python run_mc.py --draws draws_independent.csv
    python run_mc.py --compare-only

Each cell is one 288-hour solve: the twelve sample days, with the whole daily
gas and carbon series shifted by that draw's pair of numbers. Every model sees
the SAME draws, so a difference between models is the model and not the luck of
the sample - the usual common-random-numbers argument, and free here.

WHAT COMES OUT
--------------
Per draw and model, the mean DE-FR spread over the sample days and the share of
those hours in which the border separated. A hundred draws gives a distribution
of each, and the width of that distribution - not its centre - is the result:
a model whose spread barely moves when fuel prices move is a model that will
misprice anything depending on that movement.

The summary also regresses the spread on the two draws, in sigma units. Those
coefficients are the sample's own sensitivities and are directly comparable with
the per-sigma figures in section 8.2. R-squared is the linearity test: close to
one and the response is a plane, so the closed-form variance formula would have
given the same answer without any of these runs; materially below one and the
curvature is real and the Monte Carlo earned its place.

A .draw marker is written into each run directory on success, holding the pair
of shifts that produced it. A run is reused only if its marker matches the draw
being asked for now. This is the same discipline as the .ladder marker in
run_shocks.py, and for the same reason: a batch that silently reuses the results
of a different experiment fails without failing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "data" / "processed" / "runs"
BUILD = "scripts/pipeline/10_build_network.py"
DAYS_FILE = "sample_days.csv"

ARMS = {"ntc": [], "mnp": ["--max-net-pos"], "cnec": ["--flow-based"]}
BORDER = ("DE_LU", "FR")
EPS = 0.5

# Rough wall clock per 288-hour cell, from the one-day runs. Used only to warn
# before a long batch.
MINUTES = {"ntc": 1.0, "mnp": 1.3, "cnec": 2.0}


def tag_for(prefix: str, arm: str, draw: int) -> str:
    return f"{prefix}-{arm}-{draw:04d}"


def marker(tag: str) -> Path:
    return RUNS / tag / ".draw"


def existing_draw(tag: str):
    """The pair of shifts that built the run on disk, or None if unknowable."""
    m = marker(tag)
    if not m.exists():
        return None
    try:
        gas, co2 = m.read_text().split()
        return float(gas), float(co2)
    except ValueError:
        return None


def matches(tag: str, gas: float, co2: float) -> bool:
    was = existing_draw(tag)
    return was is not None and abs(was[0] - gas) < 1e-6 and abs(was[1] - co2) < 1e-6


def spread_stats(price: pd.DataFrame):
    a, b = BORDER
    if a not in price.columns or b not in price.columns:
        return float("nan"), float("nan")
    diff = (price[a] - price[b]).dropna()
    if diff.empty:
        return float("nan"), float("nan")
    return float(diff.mean()), float((diff.abs() > EPS).mean() * 100)


def collect(prefix: str, draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for d in draws.itertuples():
            path = RUNS / tag_for(prefix, arm, d.draw) / "prices.parquet"
            if not path.exists():
                continue
            mean, sep = spread_stats(pd.read_parquet(path))
            rows.append({"draw": d.draw, "arm": arm,
                         "gas_sigma": d.gas_sigma, "co2_sigma": d.co2_sigma,
                         "gas_shift": d.gas_shift, "co2_shift": d.co2_shift,
                         "mean_spread": mean, "separated_pct": sep})
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame, out: Path) -> None:
    if res.empty:
        print("\nno completed runs found")
        return
    res.to_csv(out, index=False, float_format="%.6f")

    print(f"\nDE-FR mean spread over the sample days, across "
          f"{res['draw'].nunique()} draws\n")
    print(f"  {'':<6} {'mean':>8} {'sd':>8} {'5th':>8} {'95th':>8} "
          f"{'min':>8} {'max':>8}  {'n':>4}")
    sds = {}
    for arm in ARMS:
        s = res[res.arm == arm]["mean_spread"].dropna()
        if s.empty:
            continue
        sds[arm] = float(s.std())
        print(f"  {arm:<6} {s.mean():8.2f} {s.std():8.2f} "
              f"{s.quantile(.05):8.2f} {s.quantile(.95):8.2f} "
              f"{s.min():8.2f} {s.max():8.2f}  {len(s):4d}")

    if "ntc" in sds and sds["ntc"] > 0:
        print("\n  width relative to ntc: " + "   ".join(
            f"{a} {sds[a] / sds['ntc']:.2f}x" for a in sds))

    print("\nresponse per 1 sigma of driver, from a regression on the draws")
    print("  (compare with section 8.2, which is also per sigma)\n")
    print(f"  {'':<6} {'gas':>8} {'carbon':>8} {'intercept':>10} {'R2':>7}")
    for arm in ARMS:
        s = res[res.arm == arm].dropna(subset=["mean_spread"])
        if len(s) < 5:
            continue
        X = np.column_stack([np.ones(len(s)), s.gas_sigma, s.co2_sigma])
        beta, *_ = np.linalg.lstsq(X, s.mean_spread.to_numpy(), rcond=None)
        resid = s.mean_spread.to_numpy() - X @ beta
        ss_tot = float(((s.mean_spread - s.mean_spread.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot else float("nan")
        print(f"  {arm:<6} {beta[1]:8.2f} {beta[2]:8.2f} {beta[0]:10.2f} "
              f"{r2:7.4f}")

    print("\n  R2 near 1.0 means the response is a plane over this range, so "
          "the\n  closed-form variance would have given the same answer. "
          "Materially\n  below 1.0 means real curvature, and the draws found "
          "it.")
    print(f"\n-> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", default="draws.csv")
    ap.add_argument("--days-file", default=DAYS_FILE)
    ap.add_argument("--arm", choices=list(ARMS) + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N draws, for a trial batch")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--bid-ladder", type=float, default=None)
    a = ap.parse_args()

    draws_path = ROOT / a.draws
    if not draws_path.exists():
        sys.exit(f"{draws_path} not found - run make_draws.py first")
    draws = pd.read_csv(draws_path)
    if a.limit:
        draws = draws.head(a.limit)

    prefix = "mci" if "independent" in draws_path.stem else "mc"
    arms = ARMS if a.arm == "all" else {a.arm: ARMS[a.arm]}
    out = ROOT / f"{prefix}_results.csv"

    if a.compare_only:
        summarise(collect(prefix, draws), out)
        return 0

    jobs, stale = [], 0
    for arm, extra in arms.items():
        for d in draws.itertuples():
            tag = tag_for(prefix, arm, d.draw)
            if (RUNS / tag).exists():
                if matches(tag, d.gas_shift, d.co2_shift):
                    continue
                stale += 1
            cmd = [sys.executable, BUILD, "--dates", a.days_file,
                   "--gas-shift", f"{d.gas_shift:.10f}",
                   "--co2-shift", f"{d.co2_shift:.10f}",
                   "--tag", tag] + extra
            if a.bid_ladder is not None:
                cmd += ["--bid-ladder", str(a.bid_ladder)]
            jobs.append((tag, cmd, d.gas_shift, d.co2_shift))

    total = len(draws) * len(arms)
    budget = sum(MINUTES[t.split("-")[1]] for t, _, _, _ in jobs)
    print(f"\n{a.draws}: {len(draws)} draws x {len(arms)} models = {total} cells"
          f"\n{total - len(jobs)} already on disk at this draw, "
          f"{len(jobs)} to run"
          + (f" ({stale} re-running, the draw on disk differs)" if stale else "")
          + f"\nrough estimate {budget / 60:.1f} hours")

    if a.dry_run:
        for tag, cmd, _, _ in jobs[:6]:
            print("   ", " ".join(cmd[1:]))
        if len(jobs) > 6:
            print(f"    ... and {len(jobs) - 6} more")
        return 0

    t0 = time.time()
    for i, (tag, cmd, gas, co2) in enumerate(jobs, 1):
        done = (time.time() - t0) / 60
        left = (done / (i - 1) * (len(jobs) - i + 1)) if i > 1 else float("nan")
        print(f"\n[{i}/{len(jobs)}] {tag}  gas {gas:+.2f} co2 {co2:+.2f}"
              f"   elapsed {done:.0f} min, ~{left:.0f} left", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print("  FAILED - stopping so the cause is visible")
            summarise(collect(prefix, draws), out)
            return 1
        try:
            marker(tag).write_text(f"{gas:.10f} {co2:.10f}\n", encoding="utf-8")
        except OSError as exc:
            print(f"  WARNING: could not write {marker(tag)}: {exc}")

    print(f"\ndone in {(time.time() - t0) / 60:.0f} minutes")
    summarise(collect(prefix, draws), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
