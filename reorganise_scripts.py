"""One-off: sort scripts/ into pipeline, analysis and diagnostics.

    python reorganise_scripts.py            show what would happen, change nothing
    python reorganise_scripts.py --apply    do it

Run it from the repository root. Delete it afterwards.

WHY A SCRIPT AND NOT A FEW MOVE COMMANDS
----------------------------------------
Every script resolves the repository root with

    ROOT = Path(__file__).resolve().parents[1]

which is correct while the file sits in scripts/. Moved one level deeper it
must become parents[2], or the script will silently look for data/ inside
scripts/ and find nothing. Moving the files by hand and forgetting that is the
kind of silent failure this project has been bitten by before, so the move and
the fix happen together or not at all.

Files are moved with `git mv` when the repository knows about them, so history
follows the file, and with a plain move otherwise.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

PIPELINE = [
    "01_pull_entsoe.py", "02_build_hourly.py", "03_build_inputs.py",
    "05_build_fleet.py", "06_fetch_fuel_prices.py", "07_build_fuel_prices.py",
    "10_build_network.py", "11_calibrate_links.py", "12_build_net_position.py",
    "14_calibrate_reserve.py", "20_fetch_jao_domain.py",
    "21_build_hourly_net_position.py", "22_fetch_jao_year.py",
    "29_build_domain.py", "30_fetch_fixed_terms.py", "31_close_domain.py",
]

ANALYSIS = [
    "13_price_formation.py", "15_dispatch_validation.py", "19_tune_reserve.py",
    "32_ram_reference.py", "38_shock_analysis.py", "40_shock_sizes.py",
    "41_tune_ladder.py", "42_validate_domain_year.py", "43_non_additivity.py",
    "44_sensitivity_decomposition.py",
]

DIAGNOSTICS = [
    "16_commercial_vs_physical.py", "17_zone_balance.py", "18_what_is_other.py",
    "23_probe_fc_download.py", "24_domain_schema.py", "25_domain_match_diag.py",
    "26_domain_join_key.py", "27_domain_unique_key.py", "28_domain_validate.py",
    "33_diagnose_2024_level.py", "34_where_is_the_gap.py", "35_merit_order.py",
    "37_margin_unique.py", "39_weather_diag.py",
]

GROUPS = {"pipeline": PIPELINE, "analysis": ANALYSIS, "diagnostics": DIAGNOSTICS}

DIAG_README = """# One-off diagnostics

These scripts each answered one question once, while the model was being built.
They are kept as the audit trail behind decisions recorded in RESULTS.md and
docs/FINDINGS.md; none of them is part of reproducing the results.

Nothing in the pipeline imports them. Run `python run_pipeline.py --list` from
the repository root for the scripts that matter.
"""


def git_ok() -> bool:
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                       cwd=ROOT, capture_output=True, check=True)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    if not SCRIPTS.is_dir():
        print(f"no scripts/ directory at {SCRIPTS}")
        return 2

    on_disk = {p.name for p in SCRIPTS.glob("*.py")}
    listed = {n for g in GROUPS.values() for n in g}

    unlisted = sorted(on_disk - listed)
    absent = sorted(listed - on_disk)
    if unlisted:
        print("NOT IN ANY GROUP - left where they are, decide by hand:")
        for n in unlisted:
            print("   " + n)
        print()
    if absent:
        print("LISTED BUT NOT ON DISK - ignored:")
        for n in absent:
            print("   " + n)
        print()

    use_git = git_ok()
    print(f"git repository: {'yes, using git mv' if use_git else 'no, plain move'}")
    print(f"mode: {'APPLY' if a.apply else 'dry run, nothing will change'}\n")

    moved = fixed = 0
    for group, names in GROUPS.items():
        dest = SCRIPTS / group
        present = [n for n in names if n in on_disk]
        print(f"scripts/{group}/  ({len(present)} files)")
        if a.apply:
            dest.mkdir(exist_ok=True)
        for n in present:
            src, dst = SCRIPTS / n, dest / n
            text = src.read_text(encoding="utf-8")
            needs = "parents[1]" in text
            print(f"   {n:<36}{'  + parents[1] -> parents[2]' if needs else ''}")
            if not a.apply:
                continue
            if needs:
                src.write_text(text.replace("parents[1]", "parents[2]"),
                               encoding="utf-8")
                fixed += 1
            if use_git:
                r = subprocess.run(["git", "mv", str(src.relative_to(ROOT)),
                                    str(dst.relative_to(ROOT))],
                                   cwd=ROOT, capture_output=True, text=True)
                if r.returncode != 0:      # not tracked by git yet
                    shutil.move(str(src), str(dst))
            else:
                shutil.move(str(src), str(dst))
            moved += 1
        print()

    if a.apply:
        (SCRIPTS / "diagnostics" / "README.md").write_text(DIAG_README,
                                                           encoding="utf-8")
        print(f"moved {moved} files, rewrote parents[1] in {fixed}")
        print("wrote scripts/diagnostics/README.md")
        print("\nnow run:  python run_pipeline.py --list")
        print("every stage should show without a '?'")
    else:
        print("re-run with --apply to make these changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
