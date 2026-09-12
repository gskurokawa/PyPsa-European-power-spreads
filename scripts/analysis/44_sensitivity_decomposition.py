"""Split the sensitivity difference between the two representations in two.

    python scripts/44_sensitivity_decomposition.py --year 2025
    python scripts/44_sensitivity_decomposition.py --year 2024

The perturbation study compares an estimated constant per-zone bound against
the published flow-based domain, and finds the second materially more sensitive
on DE-FR. But those two differ in TWO respects at once - estimated against
published, constant against hourly, and per zone against per element - so the
finding cannot be attributed to either.

A third arm settles it. JAO's published maxNetPos is per zone like the first
and hourly like the second, so:

    ntc  -> mnp     changes the SOURCE and the FREQUENCY of the limit
    mnp  -> cnec    changes WHAT IS LIMITED, zones to network elements
    ntc  -> cnec    the original comparison, the product of the two

Section 7 ran the same decomposition on how often each border is congested and
found the second step carries most of it. This asks whether the same holds for
the response to a driver shock, which is a different question: reproducing a
distribution and responding correctly to a change in inputs need not have the
same cause.

Sensitivity is the centred difference, (up - down) / 2, taken against each
arm's OWN base case, so the base-case difference between arms cancels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "processed" / "runs"

ARMS = {"ntc": "NTC", "mnp": "maxNetPos", "cnec": "CNEC"}
PAIRS = {"gas": ("gasup", "gasdn"), "co2": ("co2up", "co2dn"),
         "weather": ("weatherup", "weatherdn"),
         "FR nuclear": ("frnucup", "frnucdn")}
BORDERS = {"DE-FR": ("DE_LU", "FR"), "DE-PL": ("DE_LU", "PL")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    y = args.year

    out_path = ROOT / "logs" / f"sensitivity_decomposition_{y}.txt"
    out_path.parent.mkdir(exist_ok=True)
    fh = out_path.open("w", encoding="utf-8")

    def out(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        fh.write(line + "\n")

    px, missing = {}, []
    for arm in ARMS:
        for sh in ["base"] + [s for pair in PAIRS.values() for s in pair]:
            f = RUNS / f"{arm}-{y}-{sh}" / "prices.parquet"
            if f.exists():
                px[(arm, sh)] = pd.read_parquet(f)
            else:
                missing.append(f"{arm}-{y}-{sh}")
    out(f"loaded {len(px)} of {len(ARMS) * 9} runs for {y}")
    if missing:
        out(f"  MISSING: {missing}")
        out("  Arms with missing cells are reported as blank, not guessed.")
    out("")

    def sens(arm, border, driver):
        """Centred difference of the mean spread, and of the median."""
        up, dn = PAIRS[driver]
        a, b = BORDERS[border]
        need = [(arm, "base"), (arm, up), (arm, dn)]
        if any(k not in px for k in need):
            return np.nan, np.nan
        base = (px[(arm, "base")][a] - px[(arm, "base")][b]).dropna()
        u = (px[(arm, up)][a] - px[(arm, up)][b]).dropna()
        d = (px[(arm, dn)][a] - px[(arm, dn)][b]).dropna()
        mean = ((u.mean() - base.mean()) - (d.mean() - base.mean())) / 2
        med = ((u.median() - base.median()) - (d.median() - base.median())) / 2
        return mean, med

    out("=" * 104)
    out(f"1  SENSITIVITY BY ARM   {y}   EUR/MWh per the applied shock")
    out("=" * 104)
    out(f"  {'border':<8}{'driver':<13}{'NTC':>10}{'maxNetPos':>12}{'CNEC':>10}"
        f"{'mnp/ntc':>10}{'cnec/mnp':>11}{'cnec/ntc':>11}")
    rows = []
    for bname in BORDERS:
        for dname in PAIRS:
            sn, _ = sens("ntc", bname, dname)
            sm, _ = sens("mnp", bname, dname)
            sc, _ = sens("cnec", bname, dname)

            def ratio(num, den):
                return num / den if abs(den) > 1e-9 else np.nan

            r1, r2, r3 = ratio(sm, sn), ratio(sc, sm), ratio(sc, sn)
            rows.append(dict(border=bname, driver=dname, ntc=sn, mnp=sm,
                             cnec=sc, r_mnp_ntc=r1, r_cnec_mnp=r2,
                             r_cnec_ntc=r3))

            def f(v, w=10, p=2):
                return ("%*.*f" % (w, p, v)) if np.isfinite(v) else " " * (w - 1) + "-"

            out(f"  {bname:<8}{dname:<13}{f(sn)}{f(sm,12)}{f(sc)}"
                f"{f(r1)}{f(r2,11)}{f(r3,11)}")

    out("")
    out("  mnp/ntc   the effect of replacing an estimated constant bound with")
    out("            JAO's published hourly one. The limit is still per zone.")
    out("  cnec/mnp  the effect of then constraining network elements instead")
    out("            of zones. Both limits are published and hourly.")
    out("  cnec/ntc  the original two-arm comparison, the product of the two.")
    out("")
    out("  A value near 1.00 means that step changed nothing.")

    R = pd.DataFrame(rows)
    if R[["ntc", "mnp", "cnec"]].notna().all(axis=1).any():
        out("")
        out("=" * 104)
        out("2  WHICH STEP CARRIES THE DIFFERENCE")
        out("=" * 104)
        out(f"  {'border':<8}{'driver':<13}{'total gap':>12}"
            f"{'from hourly':>14}{'from elements':>16}{'hourly share':>15}")
        for _, r in R.iterrows():
            if not np.isfinite(r.ntc) or not np.isfinite(r.cnec):
                continue
            total = r.cnec - r.ntc
            step1 = r.mnp - r.ntc
            step2 = r.cnec - r.mnp
            share = (step1 / total * 100) if abs(total) > 1e-9 else np.nan
            sh = f"{share:>14.0f}%" if np.isfinite(share) else " " * 14 + "-"
            out(f"  {r.border:<8}{r.driver:<13}{total:>12.2f}"
                f"{step1:>14.2f}{step2:>16.2f}{sh}")
        out("")
        out("  Shares outside 0-100% mean the two steps moved in OPPOSITE")
        out("  directions, so neither 'carries' the difference and the split")
        out("  is not meaningful for that cell. Report those as such.")

    fh.close()
    print(f"\nwritten to logs/sensitivity_decomposition_{y}.txt")


if __name__ == "__main__":
    main()
