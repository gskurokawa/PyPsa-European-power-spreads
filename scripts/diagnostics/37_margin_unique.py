"""Is the residual dispersion the join, or the physics?

    python scripts/37_margin_unique.py

Script 32 identified the constraint formulation: published RAM, net positions
as JAO publishes them, positive sign, Switzerland excluded.  Binding
constraints came out at a median of exactly zero against a slack median of
880 MW - so the formulation is right.  But only 47% of them sat within 50 MW
of zero, and the question is what the other half are.

Two candidates, and they have different consequences:

  THE JOIN.  Script 28 measured 1.12 domain rows per (hour, element,
  contingency, direction) key.  So a third of the rows flagged as binding may
  be landing on a NEIGHBOURING CNEC - same element, a different contingency -
  whose RAM differs by roughly the margins being seen.  If so the pipeline is
  correct and only the bookkeeping is fuzzy, which does not affect the build:
  the model consumes every domain row as its own constraint and never needs
  this join at all.

  THE PHYSICS.  Something in the parse, the mapping or the net positions is
  still wrong for a subset of constraints, in which case the build would carry
  the error.

This restricts the comparison to keys that identify exactly ONE domain row,
where there is nothing to get wrong, and reports the hit rate there against
the ambiguous rows.  A large gap says the join; no gap says look further.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

LOG = open(ROOT / "logs" / "margin_unique.txt", "w", encoding="utf-8")


def out(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    LOG.write(line + "\n")


def norm(df):
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿").lower() for c in df.columns]
    return df


dom = pd.read_parquet(PROC / "fb_domain.parquet")
dom["t"] = pd.to_datetime(dom["t"], utc=True)
out(f"domain: {len(dom):,} rows, {dom['t'].nunique():,} hours")

act = norm(pd.read_csv(RAW / "jao" / "activeFbConstraints_2024.csv",
                       low_memory=False))
act["t"] = pd.to_datetime(act["datetimeutc"], utc=True, errors="coerce")
act = act[act["shadowprice"].abs() > 1e-9]
act = act[act["t"].isin(set(dom["t"]))]

# the key from script 28
dom["k"] = (dom["t"].astype("int64").astype(str) + "|" + dom["cid"])
act["k"] = (act["t"].astype("int64").astype(str) + "|"
            + act["cneceic"].astype(str) + "|"
            + act["brancheic"].astype(str) + "|"
            + act["direction"].astype(str))

counts = dom["k"].value_counts()
uniq = set(counts[counts == 1].index)
dom["unique_key"] = dom["k"].isin(uniq)
dom["bound"] = dom["k"].isin(set(act["k"]))
out(f"active rows in these hours: {len(act):,}")
out(f"domain rows flagged binding: {int(dom['bound'].sum()):,}")
out(f"  of which on an unambiguous key: "
    f"{int((dom['bound'] & dom['unique_key']).sum()):,}")

# ---- the winning formulation from script 32 -------------------------
jn = norm(pd.read_csv(RAW / "jao" / "netPos_2024.csv", low_memory=False))
jn["t"] = pd.to_datetime(jn["datetimeutc"], utc=True, errors="coerce")
jn = jn.set_index("t")
jn = jn[~jn.index.duplicated(keep="first")]

COL = {"ptdf_DE_LU": "hub_de", "ptdf_FR": "hub_fr", "ptdf_PL": "hub_pl",
       "ptdf_NL": "hub_nl", "ptdf_BE": "hub_be", "ptdf_AT": "hub_at",
       "ptdf_CZ": "hub_cz", "fixed_HR": "hub_hr", "fixed_HU": "hub_hu",
       "fixed_RO": "hub_ro", "fixed_SI": "hub_si", "fixed_SK": "hub_sk"}
have = {k: v for k, v in COL.items() if k in dom.columns and v in jn.columns}

npx = jn.reindex(pd.DatetimeIndex(dom["t"]))
lhs = np.zeros(len(dom))
for pcol, ncol in have.items():
    lhs = lhs + dom[pcol].values * np.nan_to_num(npx[ncol].values)
dom["margin"] = dom["ram"].values - lhs

# ---- the comparison --------------------------------------------------
out("\n" + "=" * 84)
out("BINDING CONSTRAINTS: unambiguous key vs ambiguous")
out("=" * 84)
out(f"  {'subset':<34}{'n':>8}{'median':>10}{'<25MW':>8}{'<50MW':>8}"
    f"{'<100MW':>9}{'<200MW':>9}")
for label, sel in [
    ("binding, key unique", dom["bound"] & dom["unique_key"]),
    ("binding, key ambiguous", dom["bound"] & ~dom["unique_key"]),
    ("binding, all", dom["bound"]),
    ("NOT binding (reference)", ~dom["bound"]),
]:
    m = dom.loc[sel, "margin"].values
    m = m[np.isfinite(m)]
    if not len(m):
        continue
    out(f"  {label:<34}{len(m):>8,}{np.median(m):>10.0f}"
        f"{100*np.mean(np.abs(m) < 25):>7.0f}%{100*np.mean(np.abs(m) < 50):>7.0f}%"
        f"{100*np.mean(np.abs(m) < 100):>8.0f}%{100*np.mean(np.abs(m) < 200):>8.0f}%")

out("\n  If 'key unique' is much tighter than 'key ambiguous', the residual")
out("  dispersion is the bookkeeping and the build is unaffected - the model")
out("  consumes every domain row directly and never performs this join.")

# ---- does the shadow price explain the rest? -------------------------
out("\n" + "=" * 84)
out("DOES A SMALL SHADOW PRICE MEAN A LOOSELY BINDING CONSTRAINT?")
out("=" * 84)
sp = act.groupby("k")["shadowprice"].max()
b = dom[dom["bound"] & dom["unique_key"]].copy()
b["sp"] = b["k"].map(sp).abs()
b = b[np.isfinite(b["margin"]) & b["sp"].notna()]
if len(b):
    qs = [0, .25, .5, .75, .9, 1.0]
    edges = b["sp"].quantile(qs).values
    out(f"  {'shadow price band':<28}{'n':>8}{'median margin':>15}{'<50MW':>8}")
    for i in range(len(edges) - 1):
        m = b[(b["sp"] >= edges[i]) & (b["sp"] <= edges[i + 1])]
        if len(m) < 5:
            continue
        out(f"  {edges[i]:>8.1f} - {edges[i+1]:<8.1f} EUR/MW"
            f"{len(m):>10,}{np.median(m['margin']):>15.0f}"
            f"{100*np.mean(np.abs(m['margin']) < 50):>7.0f}%")
    out("\n  A constraint with a near-zero shadow price is barely binding, so a")
    out("  wider margin there is expected rather than an error.")

LOG.close()
print("\nwritten to logs/margin_unique.txt")
