"""Why do only 26% of binding constraints appear in the presolved domain?

    python scripts/25_domain_match_diag.py

Three candidate explanations, tested separately rather than together:

  A. HOUR ALIGNMENT.  The downloaded files are named "0100 - 0200" while the
     column is DateTimeUtc.  If the two datasets are offset by a whole number
     of hours, the same constraint will be present in both files but never in
     the same row.  Test: for each unmatched active row, find every hour in
     the domain that does contain that constraint, and histogram the offset.
     A spike at a single non-zero offset settles it.

  B. THE MATCH KEY.  Each element appears in the domain up to six times per
     hour: DIRECT and OPPOSITE (the same wire with every PTDF sign flipped),
     each in three CRA states.  Matching on name and contingency alone is
     therefore ambiguous, and comparing PTDFs after an arbitrary
     drop_duplicates can compare a row against its own sign-flipped twin.

  C. A DIFFERENT POPULATION.  activeFbConstraints may carry rows that are not
     CNECs at all - the external/allocation constraints found in section 12 -
     which would never appear in finalComputation.
"""
from pathlib import Path

import pandas as pd

ROOT = Path("data/raw/jao")
DOM  = ROOT / "domain2024"


def norm(df):
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿").lower() for c in df.columns]
    return df


frames = []
for f in sorted(DOM.glob("fc_*.csv.gz")):
    try:
        frames.append(pd.read_csv(f, low_memory=False))
    except Exception:                                         # noqa: BLE001
        pass
dom = norm(pd.concat(frames, ignore_index=True))
dom["t"] = pd.to_datetime(dom["datetimeutc"], utc=True, errors="coerce")

act = norm(pd.read_csv(ROOT / "activeFbConstraints_2024.csv", low_memory=False))
act["t"] = pd.to_datetime(act["datetimeutc"], utc=True, errors="coerce")
act = act[act["t"].between(dom["t"].min(), dom["t"].max())]
act = act[act["shadowprice"].abs() > 1e-9]

print(f"domain {len(dom):,} rows over {dom['t'].nunique():,} hours "
      f"({dom['t'].min()} .. {dom['t'].max()})")
print(f"active {len(act):,} rows over {act['t'].nunique():,} hours "
      f"({act['t'].min()} .. {act['t'].max()})")


def elem(df, n, c):
    return (df[n].astype(str).str.strip() + " || " +
            df[c].astype(str).str.strip())


dom["e"] = elem(dom, "cnename", "contname")
act["e"] = elem(act, "cnecname", "contname")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("C  IS THE UNMATCHED POPULATION A DIFFERENT KIND OF ROW?")
print("=" * 88)
in_dom_ever = act["e"].isin(set(dom["e"]))
print(f"  active constraints whose element+contingency exists in the domain")
print(f"  AT ANY HOUR: {in_dom_ever.mean()*100:.1f} %")
if (~in_dom_ever).any():
    print("\n  the ten most common that never appear at all:")
    for e, n in act.loc[~in_dom_ever, "e"].value_counts().head(10).items():
        print(f"    {n:>5}  {e[:74]}")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("A  HOUR ALIGNMENT")
print("=" * 88)
# for constraints that DO exist in the domain, compare the hours
dh = dom.groupby("e")["t"].apply(set)
sub = act[in_dom_ever].copy()
sub = sub.sample(min(4000, len(sub)), random_state=0)

offsets = []
for e, t in zip(sub["e"], sub["t"]):
    hrs = dh.get(e)
    if not hrs:
        continue
    best = min(hrs, key=lambda h: abs((h - t).total_seconds()))
    offsets.append(round((best - t).total_seconds() / 3600))
off = pd.Series(offsets)
print(f"  nearest domain hour minus active hour, over {len(off):,} samples:")
print(off.value_counts().head(10).to_string())
print(f"\n  exact same hour: {100*(off == 0).mean():.1f} %")
if (off == 0).mean() < 0.9:
    print("  a spike away from 0 means a whole-hour offset between the two")
    print("  datasets, and the fix is a shift, not a reparse.")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("B  THE MATCH KEY: direction and CRA state")
print("=" * 88)
for col in ("direction", "cnestatus"):
    if col in dom.columns:
        print(f"\n  domain {col}: ")
        print(dom[col].value_counts().to_string())
    if col in act.columns:
        print(f"  active {col}: ")
        print(act[col].value_counts().to_string())

dom["k3"] = (dom["t"].astype("int64").astype(str) + "|" + dom["e"] + "|"
             + dom.get("direction", pd.Series("", index=dom.index)).astype(str))
act["k3"] = (act["t"].astype("int64").astype(str) + "|" + act["e"] + "|"
             + act.get("direction", pd.Series("", index=act.index)).astype(str))
dom["k2"] = dom["t"].astype("int64").astype(str) + "|" + dom["e"]
act["k2"] = act["t"].astype("int64").astype(str) + "|" + act["e"]

print(f"\n  match on hour+element+contingency          : "
      f"{act['k2'].isin(set(dom['k2'])).mean()*100:5.1f} %")
print(f"  match on hour+element+contingency+direction: "
      f"{act['k3'].isin(set(dom['k3'])).mean()*100:5.1f} %")
print(f"  domain rows per (hour, element, contingency): "
      f"{len(dom)/dom['k2'].nunique():.2f}")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("ONE HOUR, SIDE BY SIDE")
print("=" * 88)
h = act["t"].value_counts().idxmax()
a1 = act[act["t"] == h]
d1 = dom[dom["t"] == h]
print(f"  {h}   active {len(a1)} rows, domain {len(d1)} rows\n")
print("  ACTIVE (what bound):")
for _, r in a1.iterrows():
    print(f"    sp {r['shadowprice']:>9.2f}  {str(r['e'])[:66]}")
print("\n  DOMAIN, the 12 with the smallest margin (ram is what is left):")
if "ram" in d1.columns:
    for _, r in d1.nsmallest(12, "ram").iterrows():
        print(f"    ram {r['ram']:>8.1f}  {str(r['direction'])[:8]:<8} "
              f"{str(r.get('cnestatus',''))[:10]:<10} {str(r['e'])[:52]}")
