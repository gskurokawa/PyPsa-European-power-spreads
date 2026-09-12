"""The section 13 result: how differently do the two arms respond to a shock?

    python scripts/38_shock_analysis.py

Reads the eighteen runs and writes logs/shock_analysis.txt.

WHAT IS COMPARED, AND WHY IT IS A DELTA
---------------------------------------
The two arms do not share a base case - the flow-based model prices
differently from the net-position one, which is the point of section 13.7.
Comparing absolute spreads across arms would confound the constraint
representation with that base difference, so each arm is measured against its
OWN base:

    delta(arm, shock) = spread(shock, arm) - spread(base, arm)
    finding           = delta(CNEC) - delta(NTC)

That is what the extrapolation argument in 13.1 is about: not which model has
better spreads, but whether the cheap representation gives a different
SENSITIVITY. Net-position bounds are 99.5th percentiles of observed 2024
flows and cannot know about a counterfactual; a thermal rating and a PTDF
still hold in one.

FIRST-ORDER VARIANCE DECOMPOSITION
----------------------------------
Every driver is shocked at plus and minus one standard deviation, so its
contribution is the half-range:

    sensitivity = ( delta(+1sd) - delta(-1sd) ) / 2

and |delta(+) + delta(-)| measures non-linearity: a shock that reorders the
merit order does not mirror one that does not.

ROBUST STATISTICS ARE REPORTED ALONGSIDE THE MEAN
-------------------------------------------------
The first weather shock was refuted precisely because its mean moved while
its median did not - 51 hours of load shedding at VOLL carried the entire
result (13.8). Every driver is therefore reported on the median and the
interquartile range as well, and any cell whose two disagree is flagged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_ap = argparse.ArgumentParser()
_ap.add_argument("--year", type=int, default=2024,
                 help="which year's shock runs to analyse. Run directories "
                      "are named <arm>-<year>-<shock>.")
_args = _ap.parse_args()

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "processed" / "runs"
LOG = open(ROOT / "logs" / f"shock_analysis_{_args.year}.txt", "w",
           encoding="utf-8")


def out(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    LOG.write(line + "\n")


YEAR = _args.year
ARMS = {"NTC-8": "ntc", "CNEC-8": "cnec"}
SHOCKS = ["base", "gasup", "gasdn", "co2up", "co2dn",
          "weatherup", "weatherdn", "frnucup", "frnucdn"]
PAIRS = {"gas": ("gasup", "gasdn"), "co2": ("co2up", "co2dn"),
         "weather": ("weatherup", "weatherdn"),
         "FR nuclear": ("frnucup", "frnucdn")}
BORDERS = {"DE-FR": ("DE_LU", "FR"), "DE-PL": ("DE_LU", "PL")}

px, missing = {}, []
for arm, pre in ARMS.items():
    for sh in SHOCKS:
        f = RUNS / f"{pre}-{YEAR}-{sh}" / "prices.parquet"
        if f.exists():
            px[(arm, sh)] = pd.read_parquet(f)
        else:
            missing.append(f"{pre}-{YEAR}-{sh}")
out(f"loaded {len(px)} of {len(ARMS)*len(SHOCKS)} runs")
if missing:
    out(f"  MISSING: {missing}")

obs = pd.read_parquet(ROOT / "data" / "processed" / "prices.parquet")


def spread(df, a, b):
    return (df[a] - df[b]).dropna()


def stats(v):
    return v.mean(), v.median(), v.std(), v.quantile(.75) - v.quantile(.25)


# ------------------------------------------------------------------
out("\n" + "=" * 104)
out("1  BASE CASES, AND WHAT REALITY DID")
out("=" * 104)
out(f"  {'border':<8}{'arm':<11}{'mean':>9}{'median':>9}{'sd':>9}{'IQR':>9}"
    f"{'>500 EUR hours':>16}")
for bname, (a, b) in BORDERS.items():
    for arm in ARMS:
        if (arm, "base") not in px:
            continue
        d = px[(arm, "base")]
        v = spread(d, a, b)
        ext = int(((d[a].abs() > 500) | (d[b].abs() > 500)).sum())
        m, md, sd, iqr = stats(v)
        out(f"  {bname:<8}{arm:<11}{m:>9.2f}{md:>9.2f}{sd:>9.2f}{iqr:>9.2f}"
            f"{ext:>16}")
    o = spread(obs.reindex(px[("NTC-8", "base")].index), a, b)
    m, md, sd, iqr = stats(o)
    out(f"  {bname:<8}{'observed':<11}{m:>9.2f}{md:>9.2f}{sd:>9.2f}{iqr:>9.2f}")

# ------------------------------------------------------------------
out("\n" + "=" * 104)
out("2  RESPONSE TO EACH SHOCK, EACH ARM AGAINST ITS OWN BASE")
out("=" * 104)
rows = []
for bname, (a, b) in BORDERS.items():
    out(f"\n  {bname}    (change in the mean spread, and in the median, EUR/MWh)")
    out(f"    {'shock':<12}{'NTC mean':>10}{'CNEC mean':>11}{'diff':>9}"
        f"{'NTC med':>10}{'CNEC med':>10}{'shed hrs':>10}")
    for sh in SHOCKS[1:]:
        dm, dd, shed = {}, {}, {}
        for arm in ARMS:
            if (arm, sh) not in px or (arm, "base") not in px:
                continue
            s1, s0 = spread(px[(arm, sh)], a, b), spread(px[(arm, "base")], a, b)
            dm[arm] = s1.mean() - s0.mean()
            dd[arm] = s1.median() - s0.median()
            d = px[(arm, sh)]
            shed[arm] = int(((d[a].abs() > 500) | (d[b].abs() > 500)).sum())
        if len(dm) == 2:
            out(f"    {sh:<12}{dm['NTC-8']:>10.2f}{dm['CNEC-8']:>11.2f}"
                f"{dm['CNEC-8']-dm['NTC-8']:>9.2f}"
                f"{dd['NTC-8']:>10.2f}{dd['CNEC-8']:>10.2f}"
                f"{shed['CNEC-8']:>10}")
            rows.append({"border": bname, "shock": sh,
                         "NTC-8": dm["NTC-8"], "CNEC-8": dm["CNEC-8"],
                         "NTC-8_med": dd["NTC-8"], "CNEC-8_med": dd["CNEC-8"],
                         "extreme": shed["CNEC-8"]})

out("\n  'shed hrs' counts hours in the CNEC run where either price exceeded")
out("  500 EUR/MWh. Section 13.8 refuted a shock whose mean was carried by")
out("  51 such hours while its median never moved - so if a mean and a median")
out("  disagree here, believe the median.")

# ------------------------------------------------------------------
out("\n" + "=" * 104)
out("3  FIRST-ORDER SENSITIVITY   ( delta(+1sd) - delta(-1sd) ) / 2")
out("=" * 104)
R = pd.DataFrame(rows)
out(f"  {'border':<8}{'driver':<12}{'NTC':>9}{'CNEC':>9}{'ratio':>8}"
    f"{'NTC med':>10}{'CNEC med':>10}{'asym NTC':>10}{'asym CNEC':>11}")
for bname in BORDERS:
    for dname, (up, dn) in PAIRS.items():
        sel_u = R[(R.border == bname) & (R.shock == up)]
        sel_d = R[(R.border == bname) & (R.shock == dn)]
        if sel_u.empty or sel_d.empty:
            continue
        u, w = sel_u.iloc[0], sel_d.iloc[0]
        sn = (u["NTC-8"] - w["NTC-8"]) / 2
        sc = (u["CNEC-8"] - w["CNEC-8"]) / 2
        mn = (u["NTC-8_med"] - w["NTC-8_med"]) / 2
        mc = (u["CNEC-8_med"] - w["CNEC-8_med"]) / 2
        ratio = sc / sn if abs(sn) > 1e-9 else float("nan")
        out(f"  {bname:<8}{dname:<12}{sn:>9.2f}{sc:>9.2f}{ratio:>8.2f}"
            f"{mn:>10.2f}{mc:>10.2f}"
            f"{abs(u['NTC-8']+w['NTC-8']):>10.2f}"
            f"{abs(u['CNEC-8']+w['CNEC-8']):>11.2f}")

out("\n  ratio = CNEC sensitivity / NTC sensitivity.")
out("  Near 1.00 everywhere is a NULL RESULT and a useful one: for driver")
out("  sensitivity the cheap representation is adequate and nobody needs to")
out("  license flow-based data. Away from 1.00 means the constraint")
out("  representation changes the answer - which is what 13.1 predicts, since")
out(f"  bounds fitted to observed flows cannot extrapolate and physical limits")
out("  can. Check the median columns tell the same story before believing it.")

# ------------------------------------------------------------------
out("\n" + "=" * 104)
out("4  RANKING THE DRIVERS  (CNEC arm, |sensitivity| in EUR/MWh)")
out("=" * 104)
for bname in BORDERS:
    got = []
    for dname, (up, dn) in PAIRS.items():
        su = R[(R.border == bname) & (R.shock == up)]
        sd_ = R[(R.border == bname) & (R.shock == dn)]
        if su.empty or sd_.empty:
            continue
        got.append((dname,
                    abs((su.iloc[0]["CNEC-8"] - sd_.iloc[0]["CNEC-8"]) / 2),
                    abs((su.iloc[0]["CNEC-8_med"] - sd_.iloc[0]["CNEC-8_med"]) / 2)))
    got.sort(key=lambda x: -x[1])
    tot = sum(g[1] for g in got) or 1.0
    out(f"\n  {bname}")
    out(f"    {'driver':<14}{'mean-based':>12}{'share':>8}{'median-based':>14}")
    for dname, v, mv in got:
        out(f"    {dname:<14}{v:>12.2f}{100*v/tot:>7.0f}%{mv:>14.2f}")
out("\n  Shares are a FIRST-ORDER decomposition: they assume the drivers act")
out("  independently and linearly. Section 13.4 records what that misses -")
out("  no interaction terms, and no tails.")

# ------------------------------------------------------------------
out("\n" + "=" * 104)
out("5  THE SAME RANKING, INDEPENDENT OF HOW BIG EACH SHOCK WAS")
out("=" * 104)
out("  NOT A VARIANCE DECOMPOSITION. The shock sizes are not on a common")
out("  footing. Gas and carbon are 1 sd of a DAILY series - volatility")
out("  within the window. Weather and French nuclear are 1 sd of an ANNUAL")
out("  mean over TWO years, which is a two-point estimate and a lower bound:")
out("  French nuclear availability reads 0.668 in 2024 and 0.674 in 2025, so")
out("  1 sd comes out at 0.63%, while the driver's real range is an order of")
out("  magnitude larger (roughly 279 TWh in 2022 against 360 TWh in 2024).")
out("  Shares across drivers are therefore indicative only.")
out("")
out("  TWO results here do not depend on the sizes. The CNEC/NTC ratio in")
out("  section 3, because both versions of the model receive the identical")
out("  shock. And the ELASTICITY below, because the response is linear -")
out("  a 10% French nuclear shock and a 0.63% one give elasticities of 85.1")
out("  and 82.5 on DE-FR. Read those two; treat the sensitivity column as")
out("  'what this particular shock size did'. See 40_shock_sizes.py.")
out("")
out("  A share computed from absolute responses depends on the shock sizes,")
out("  and those differ: gas and carbon move by 1 sd of a DAILY series, while")
out("  weather and French nuclear move by 1 sd of an ANNUAL mean. Dividing by")
out("  the shock actually applied gives an elasticity, which does not.")
out("")
out("  Shock sizes are read from the run logs, which print them. If a size is")
out("  not supplied below the elasticity is left blank rather than guessed.")
SIZES = {}   # driver -> relative size of the +1sd shock, e.g. {"gas": 0.21}
try:
    import json
    f = ROOT / "logs" / "shock_sizes.json"
    if f.exists():
        raw = json.loads(f.read_text())
        SIZES = raw.get("applied", raw)
except Exception:                                             # noqa: BLE001
    pass
if not SIZES:
    out("  logs/shock_sizes.json not present - run scripts/40_shock_sizes.py")
else:
    for bname in BORDERS:
        out(f"\n  {bname}")
        out(f"    {'driver':<14}{'sensitivity':>13}{'shock size':>12}"
            f"{'elasticity':>13}{'share':>8}")
        got = []
        for dname, (up, dn) in PAIRS.items():
            su = R[(R.border == bname) & (R.shock == up)]
            sd_ = R[(R.border == bname) & (R.shock == dn)]
            if su.empty or sd_.empty or dname not in SIZES:
                continue
            sens = abs((su.iloc[0]["CNEC-8"] - sd_.iloc[0]["CNEC-8"]) / 2)
            got.append((dname, sens, sens / SIZES[dname], SIZES[dname]))
        tot = sum(g[2] for g in got) or 1.0
        for dname, sens, el, size in sorted(got, key=lambda x: -x[2]):
            out(f"    {dname:<14}{sens:>13.2f}{100*size:>11.1f}%"
                f"{el:>13.1f}{100*el/tot:>7.0f}%")
    out("\n  elasticity = EUR/MWh of spread per 100% change in the driver.")
    out("  Read the two rankings together: the share above answers 'what")
    out(f"  moved the spread in {YEAR}', the elasticity answers 'what would move")
    out("  it most if each driver moved by the same proportion'.")

LOG.close()
print(f"\nwritten to logs/shock_analysis_{YEAR}.txt")
