"""Run the pipeline end to end, or any contiguous part of it.

    python run_pipeline.py --list                  what the stages are
    python run_pipeline.py --year 2025             everything, for 2025
    python run_pipeline.py --year 2025 --from 10   skip the data pull
    python run_pipeline.py --year 2025 --to 07     inputs only
    python run_pipeline.py --year 2025 --dry-run   print the commands, run nothing

Stages run in the listed order and the run stops at the first failure, so a
half-built set of inputs is never handed to the next stage.

WHY THIS FILE EXISTS
--------------------
The scripts folder holds the stages, the analyses that produce the numbers in
RESULTS.md, and a set of one-off diagnostics. Nothing in the folder said which
of those a person has to run, or in what order. This does.

Script paths are resolved by searching scripts/ for the filename, so moving a
stage into scripts/pipeline/ or scripts/analysis/ does not break the runner.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

# (number, filename, what it does, extra args)
# {year} is substituted from --year. A stage with no year argument ignores it.
STAGES = [
    ("01", "01_pull_entsoe.py", "pull ENTSO-E load, prices, generation, flows, outages", []),
    ("02", "02_build_hourly.py", "resample the raw pulls to one hourly frame", []),
    ("03", "03_build_inputs.py", "assemble model inputs from the hourly frames", []),
    ("05", "05_build_fleet.py", "build the unit-level thermal fleet", []),
    ("06", "06_fetch_fuel_prices.py", "fetch gas, coal and carbon price series", []),
    ("07", "07_build_fuel_prices.py", "build the daily fuel and carbon frame", []),
    ("11", "11_calibrate_links.py", "derive per-link ratings from observed exchange", []),
    ("12", "12_build_net_position.py", "estimate the per-zone net-position bounds (NTC model)", []),
    ("14", "14_calibrate_reserve.py", "calibrate the operating-reserve requirement", []),
    ("20", "20_fetch_jao_domain.py", "fetch the JAO flow-based domain", []),
    ("21", "21_build_hourly_net_position.py", "build hourly observed net positions", []),
    ("22", "22_fetch_jao_year.py", "fetch a full year of JAO series", ["--year", "{year}"]),
    ("29", "29_build_domain.py", "consolidate the raw domain into one table", []),
    ("30", "30_fetch_fixed_terms.py", "fetch the fixed terms the domain needs", []),
    ("31", "31_close_domain.py", "close gaps in the domain and write the final table", []),
    ("10", "10_build_network.py", "build and solve the model (add --flow-based or --max-net-pos)",
     ["--year", "{year}", "--chunk-days", "30", "--bid-ladder", "20"]),
]


def find(name: str) -> Path | None:
    """Locate a script anywhere under scripts/, so subfolders do not break this."""
    hits = sorted(SCRIPTS.rglob(name))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--from", dest="start", default=None,
                    help="first stage number to run, e.g. 10")
    ap.add_argument("--to", dest="stop", default=None,
                    help="last stage number to run, e.g. 07")
    ap.add_argument("--list", action="store_true", help="print the stages and exit")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.list:
        print(f"{'':2} {'script':<34} what it does")
        for num, name, what, _ in STAGES:
            mark = " " if find(name) else "?"
            print(f"{mark}{num} {name:<34} {what}")
        print("\n? = not found under scripts/")
        return 0

    order = [s[0] for s in STAGES]
    lo = order.index(a.start) if a.start in order else 0
    hi = order.index(a.stop) + 1 if a.stop in order else len(STAGES)
    if a.start and a.start not in order:
        print(f"--from {a.start} is not a stage number; see --list")
        return 2
    if a.stop and a.stop not in order:
        print(f"--to {a.stop} is not a stage number; see --list")
        return 2
    todo = STAGES[lo:hi]

    missing = [n for _, n, _, _ in todo if not find(n)]
    if missing:
        print("these stages are not under scripts/ - aborting before anything runs:")
        for m in missing:
            print("   " + m)
        return 2

    print(f"year {a.year}   {len(todo)} stages   "
          f"{todo[0][0]} to {todo[-1][0]}")
    t0 = time.time()
    for i, (num, name, what, extra) in enumerate(todo, 1):
        path = find(name)
        args = [x.format(year=a.year) for x in extra]
        cmd = [sys.executable, str(path.relative_to(ROOT))] + args
        print(f"\n{'='*70}\n[{i}/{len(todo)}] {num}  {what}\n"
              f"    {' '.join(cmd)}\n{'='*70}", flush=True)
        if a.dry_run:
            continue
        t1 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        print(f"  -> exit {r.returncode} in {(time.time()-t1)/60:.1f} min", flush=True)
        if r.returncode != 0:
            print(f"\nSTOPPED at stage {num}. Nothing after it has run.")
            return r.returncode
    print(f"\ndone in {(time.time()-t0)/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
