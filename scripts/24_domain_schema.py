"""What is actually in the presolved domain, and does it contain reality?

    python scripts/24_domain_schema.py

Reads whatever scripts/22_fetch_jao_year.py has put in data/raw/jao/domain2024
so far - it does not need the download to have finished - and answers four
questions before any modelling starts.

WHY THIS TEST FIRST
-------------------
The domain (`finalComputation`, presolved) says which constraints the market
coupling HAD to respect.  activeFbConstraints says which ones actually BOUND,
with their shadow prices.  The second is a subset of the first, by definition.

That gives a correctness test that needs no net positions, no LP and no model:
every constraint JAO reports as active in a given hour must be present in the
presolved domain for that same hour, and its PTDF row must be the same in both
files.  If that holds, the parse, the hub mapping and the hour alignment are
all correct.  If it does not, exactly one of those is wrong and the mismatch
says which.

It also answers the question the build's cost depends on: how many DISTINCT
constraints are there, and how concentrated is the binding?
"""
from pathlib import Path
import re

import pandas as pd

ROOT = Path("data/raw/jao")
DOM  = ROOT / "domain2024"

ZONES = ["AT", "BE", "CZ", "DE", "FR", "HR", "HU",
         "NL", "PL", "RO", "SI", "SK"]


def norm(df):
    """lowercase column names so the two files can be compared."""
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿").lower() for c in df.columns]
    return df


# ==================================================================
print("=" * 90)
print("1  WHAT IS ON DISK")
print("=" * 90)
files = sorted(DOM.glob("fc_*.csv.gz"))
if not files:
    raise SystemExit("  nothing downloaded yet - let script 22 run first")
print(f"  {len(files)} two-day chunks, "
      f"{sum(f.stat().st_size for f in files)/1e6:.0f} MB")
print(f"  {files[0].name} .. {files[-1].name}")

# Script 22 may still be running and writes each chunk straight to its final
# name, so the newest file can be half-written.  Skip anything that will not
# read rather than dying on a truncated gzip.
frames, skipped = [], []
for f in files:
    try:
        frames.append(pd.read_csv(f, low_memory=False))
    except Exception as exc:                                  # noqa: BLE001
        skipped.append((f.name, type(exc).__name__))
if skipped:
    print(f"  skipped {len(skipped)} file(s) still being written: "
          f"{[n for n, _ in skipped]}")
if not frames:
    raise SystemExit("  no complete chunks yet - try again in a minute")
dom = norm(pd.concat(frames, ignore_index=True))
tcol = next(c for c in dom.columns if "datetime" in c)
dom[tcol] = pd.to_datetime(dom[tcol], utc=True, errors="coerce")
dom = dom.rename(columns={tcol: "t"})
print(f"  {len(dom):,} rows, {dom['t'].nunique():,} distinct hours, "
      f"{len(dom)/max(dom['t'].nunique(),1):.0f} rows per hour")

print("\n  every column:")
for i in range(0, len(dom.columns), 4):
    print("     " + "  ".join(f"{c:<24}" for c in dom.columns[i:i + 4]))


# ==================================================================
print("\n" + "=" * 90)
print("2  WHICH COLUMNS ARE THE PTDFs, AND WHICH IS THE MARGIN")
print("=" * 90)
ptdf = {}
for z in ZONES:
    hits = [c for c in dom.columns
            if re.fullmatch(rf"(hub_|ptdf_?)?{z.lower()}", c)
            or c.endswith(f"_{z.lower()}") and "ptdf" in c
            or c == f"ptdf{z.lower()}"]
    if hits:
        ptdf[z] = hits[0]
print(f"  zone PTDF columns found: {len(ptdf)} of {len(ZONES)}")
print(f"    {ptdf}")
missing = [z for z in ZONES if z not in ptdf]
if missing:
    print(f"    NOT FOUND: {missing}   <- fix the pattern before going on")

virtual = [c for c in dom.columns
           if c.startswith(("hub_", "ptdf")) and c not in ptdf.values()]
print(f"\n  other hub-like columns (virtual hubs / external borders): "
      f"{len(virtual)}")
print(f"    {virtual}")

for want in ["ram", "fmax", "fref", "frm", "presolved", "direction",
             "cnestatus", "minramfactor"]:
    hits = [c for c in dom.columns if want in c]
    if hits:
        d = dom[hits[0]]
        if pd.api.types.is_numeric_dtype(d):
            print(f"  {hits[0]:<22} numeric  mean {d.mean():>10.1f}  "
                  f"min {d.min():>10.1f}  max {d.max():>10.1f}")
        else:
            print(f"  {hits[0]:<22} {d.nunique()} distinct: "
                  f"{list(d.dropna().unique()[:6])}")


# ==================================================================
print("\n" + "=" * 90)
print("3  DOES THE DOMAIN CONTAIN EVERY CONSTRAINT THAT ACTUALLY BOUND?")
print("=" * 90)
act = norm(pd.read_csv(ROOT / "activeFbConstraints_2024.csv", low_memory=False))
act["t"] = pd.to_datetime(act["datetimeutc"], utc=True, errors="coerce")
act = act[act["t"].isin(set(dom["t"]))]
act = act[act["shadowprice"].abs() > 1e-9]
print(f"  active rows in the same hours: {len(act):,}")

# name columns differ between the two files: cnecName vs cneName
dn = next((c for c in dom.columns if c in ("cnename", "cnecname")), None)
an = next((c for c in act.columns if c in ("cnecname", "cnename")), None)
dc = next((c for c in dom.columns if "contname" in c), None)
ac = next((c for c in act.columns if "contname" in c), None)
print(f"  matching on: domain[{dn}, {dc}]  vs  active[{an}, {ac}]")

def key(df, n, c):
    return (df["t"].astype("int64").astype(str) + "|" +
            df[n].astype(str).str.strip() + "|" +
            (df[c].astype(str).str.strip() if c else ""))

dom["k"], act["k"] = key(dom, dn, dc), key(act, an, ac)
found = act["k"].isin(set(dom["k"]))
print(f"\n  active constraints present in the presolved domain: "
      f"{found.mean()*100:.2f} %  ({int(found.sum()):,} of {len(found):,})")
if found.mean() < 0.999:
    print("  MISSING EXAMPLES (name | contingency | hour | shadow price):")
    for _, r in act[~found].head(8).iterrows():
        print(f"    {str(r[an])[:38]:<38} | {str(r[ac])[:26]:<26} | "
              f"{r['t']} | {r['shadowprice']:>10.1f}")
    print("  a low number here means the hour alignment or the name columns")
    print("  do not line up - not that JAO's presolve is dropping binders.")


# ==================================================================
print("\n" + "=" * 90)
print("4  DO THE PTDFs AGREE BETWEEN THE TWO FILES?")
print("=" * 90)
if found.any() and ptdf:
    m = act[found].merge(
        dom[["k"] + [ptdf[z] for z in ptdf]].drop_duplicates("k"),
        on="k", how="left", suffixes=("_act", "_dom"))
    worst = 0.0
    for z, col in ptdf.items():
        a_col = f"hub_{z.lower()}"
        if a_col not in m.columns:
            continue
        d = (pd.to_numeric(m[a_col], errors="coerce")
             - pd.to_numeric(m[col if col in m.columns else col + "_dom"],
                             errors="coerce")).abs()
        worst = max(worst, d.max() if d.notna().any() else 0)
        print(f"    {z}: max |difference| {d.max():.6f}   "
              f"mean {d.mean():.6f}   compared on {int(d.notna().sum()):,} rows")
    print(f"\n  worst disagreement anywhere: {worst:.6f}")
    print("  ~0 means the hub mapping and the sign convention are the same in")
    print("  both files, and the domain can be trusted as the model's input.")


# ==================================================================
print("\n" + "=" * 90)
print("5  HOW MANY DISTINCT CONSTRAINTS, AND HOW CONCENTRATED IS BINDING?")
print("=" * 90)
dom["cid"] = dom[dn].astype(str).str.strip() + " || " + \
             (dom[dc].astype(str).str.strip() if dc else "")
act["cid"] = act[an].astype(str).str.strip() + " || " + \
             (act[ac].astype(str).str.strip() if ac else "")
print(f"  distinct constraints in the domain : {dom['cid'].nunique():,}")
print(f"  distinct constraints that ever bind: {act['cid'].nunique():,}")

freq = act["cid"].value_counts()
hours = act["t"].nunique()
print(f"\n  binding hours are carried by very few constraints:")
for n in (10, 20, 50, 100, 200):
    if n <= len(freq):
        print(f"    top {n:>3} constraints cover "
              f"{100*freq.head(n).sum()/len(act):>5.1f} % of all binding rows")
print("\n  the ten most frequent:")
for cid, n in freq.head(10).items():
    print(f"    {n:>5} hours ({100*n/hours:>4.1f}%)  {cid[:78]}")
print("\n  this is the number the build's solve cost turns on: if the top 200")
print("  cover nearly everything, the LP can carry 200 constraints an hour")
print("  instead of every presolved row.")
