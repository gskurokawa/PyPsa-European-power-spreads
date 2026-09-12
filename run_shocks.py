"""Run the eighteen cells of the FINDINGS section 13 study.

    python run_shocks.py --year 2025 --bid-ladder 20
    python run_shocks.py --year 2025 --bid-ladder 20 --arm ntc
    python run_shocks.py --year 2025 --bid-ladder 20 --arm mnp
    python run_shocks.py --year 2025 --bid-ladder 20 --dry-run

Three versions of the model - estimated net-position limits, JAO's published
hourly net-position limits, and the published flow-based domain - each crossed
with a base case and eight driver shocks.  Everything
else is held identical, so the finding is the DIFFERENCE in response between
the two:

    delta(version) = spread(shock, version) - spread(base, version)
    finding        = delta(CNEC) - delta(NTC)

Comparing absolute levels across versions would confound the constraint
representation with the base-case difference between them; comparing each
version's response against its OWN base does not.

WHY THE LADDER IS A FLAG AND NOT A YAML EDIT
--------------------------------------------
An earlier 2025 batch - all eighteen cells, two and a half hours - ran with
the wrong nuclear bid ladder because the instruction to edit
config/technology.yaml first was not carried out, and the pre-existing base
directories were silently skipped.  Nothing failed.  The runs completed, the
analysis ran, and the numbers were quietly the wrong experiment.

That is the same failure mode section 6.6 records, so the parameter now
travels with the command:

  * --bid-ladder is passed straight through to 10_build_network.py
  * the effective value is written to a .ladder marker inside each run
    directory when it completes
  * a run is only skipped if its marker MATCHES the ladder being requested;
    a mismatch is re-run, and the reason is printed

Runs are otherwise skipped if their output directory already exists, so this
stays restartable.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "data" / "processed" / "runs"
SHOCKS = [None, "gas+", "gas-", "co2+", "co2-",
          "weather+", "weather-", "frnuc+", "frnuc-"]

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2024)
ap.add_argument("--arm", choices=["ntc", "cnec", "mnp", "all", "both"],
                default="both",
                help="ntc: estimated constant net-position bounds. cnec: the "
                     "published flow-based domain. mnp: JAO's published hourly "
                     "net-position limits, the variant that separates the two "
                     "differences between the other arms. both: ntc+cnec "
                     "(the original pair). all: every arm.")
ap.add_argument("--bid-ladder", type=float, default=None,
                help="nuclear bid ladder in EUR/MWh, passed to every cell. "
                     "Omit to use whatever config/technology.yaml holds.")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

# The EFFECTIVE ladder - what the runs will actually use - whether it came
# from the flag or from the config. The marker records this, so a config edit
# is detected too, not just a missing flag.
cfg = yaml.safe_load(open(ROOT / "config" / "technology.yaml", encoding="utf-8"))
cfg_ladder = float((cfg.get("bid_ladder") or {}).get("Nuclear", 0.0))
effective = a.bid_ladder if a.bid_ladder is not None else cfg_ladder
source = "--bid-ladder" if a.bid_ladder is not None else "config/technology.yaml"

print(f"year {a.year}   nuclear bid ladder {effective:g} EUR/MWh  (from {source})")
if a.bid_ladder is not None and a.bid_ladder != cfg_ladder:
    print(f"  NOTE: the config holds {cfg_ladder:g}; the flag overrides it "
          f"for every cell in this batch.")

ALL_ARMS = {"ntc": [], "cnec": ["--flow-based"], "mnp": ["--max-net-pos"]}
if a.arm == "all":
    arms = dict(ALL_ARMS)
elif a.arm == "both":
    arms = {"ntc": ALL_ARMS["ntc"], "cnec": ALL_ARMS["cnec"]}
else:
    arms = {a.arm: ALL_ARMS[a.arm]}

jobs = []
for arm, extra in arms.items():
    for sh in SHOCKS:
        tag = f"{arm}-{a.year}-" + (sh.replace("+", "up").replace("-", "dn")
                                    if sh else "base")
        cmd = [sys.executable, "scripts/pipeline/10_build_network.py",
               "--year", str(a.year), "--chunk-days", "30",
               "--tag", tag] + extra
        if a.bid_ladder is not None:
            cmd += ["--bid-ladder", str(a.bid_ladder)]
        if sh:
            cmd += ["--shock", sh]
        jobs.append((tag, cmd))


def marker(tag):
    return RUNS / tag / ".ladder"


def existing_ladder(tag):
    """What ladder built the run already on disk, or None if unknowable."""
    m = marker(tag)
    if not m.exists():
        return None
    try:
        return float(m.read_text().strip())
    except ValueError:
        return None


todo, skipped, stale = [], [], []
for t, c in jobs:
    if not (RUNS / t).exists():
        todo.append((t, c))
        continue
    was = existing_ladder(t)
    if was is None:
        stale.append((t, "no .ladder marker - built before this check existed"))
        todo.append((t, c))
    elif was != effective:
        stale.append((t, f"built with ladder {was:g}, now {effective:g}"))
        todo.append((t, c))
    else:
        skipped.append(t)

print(f"\n{len(jobs)} cells, {len(skipped)} already on disk at this ladder, "
      f"{len(todo)} to run")
stale_tags = {t for t, _ in stale}
fresh = [t for t, _ in todo if t not in stale_tags]
if stale:
    print("\n  RE-RUNNING - a run exists but is not this experiment:")
    for t, why in stale:
        print(f"    {t:<22} {why}")
if fresh:
    print("\n  NEW - nothing on disk yet:")
    for t in fresh:
        print(f"    {t}")
if a.dry_run or not todo:
    sys.exit(0)

t0 = time.time()
for i, (tag, cmd) in enumerate(todo, 1):
    print(f"\n{'='*70}\n[{i}/{len(todo)}] {tag}   ladder {effective:g}\n{'='*70}",
          flush=True)
    t1 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"  -> exit {r.returncode} in {(time.time()-t1)/60:.1f} min "
          f"| elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    if r.returncode != 0:
        print("  FAILED - stopping so the cause is visible")
        break
    # Written only on success, so an interrupted run is re-done rather than
    # inheriting a marker it did not earn.
    try:
        marker(tag).write_text(f"{effective:g}\n", encoding="utf-8")
    except OSError as exc:
        print(f"  WARNING: could not write {marker(tag)}: {exc}")
print(f"\ndone in {(time.time()-t0)/60:.1f} minutes")
