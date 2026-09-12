"""Find the join key that reliably links the domain to what actually bound.

    python scripts/26_domain_join_key.py

Script 25 showed the two files describe the SAME constraints but format the
contingency name differently - activeFbConstraints writes "N-1 Gyor -
Neusiedl" where finalComputation writes "Gyor - Neusiedl Neusiedl", and the
whitespace differs too.  Joining on prose was the bug.

Both files carry EIC codes - machine identifiers for network elements - so
this tries every candidate key and reports which one actually works.  The
winner becomes the join the preprocessing uses permanently.

It also separates out the rows that CANNOT match by construction: the
external / allocation constraints found in section 12 of FINDINGS.md are not
CNECs and never appear in finalComputation, so they cap the achievable rate.
"""
from pathlib import Path
import re

import pandas as pd

ROOT = Path("data/raw/jao")
DOM  = ROOT / "domain2024"


def norm(df):
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿").lower() for c in df.columns]
    return df


def tidy(s):
    """lowercase, drop an N-1 / N-2 prefix, collapse all whitespace."""
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace(r"^n-\d+\s*", "", regex=True)
    return s.str.replace(r"\s+", "", regex=True)


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

print(f"domain {len(dom):,} rows / {dom['t'].nunique():,} hours")
print(f"active {len(act):,} rows / {act['t'].nunique():,} hours")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("0  WHICH ACTIVE ROWS COULD NEVER MATCH ANYWAY")
print("=" * 88)
namecol = "cnecname" if "cnecname" in act.columns else "cnename"
is_ext = act[namecol].isna() | act[namecol].astype(str).str.contains(
    "external", case=False, na=False)
print(f"  external / allocation rows (not CNECs): {int(is_ext.sum()):,} "
      f"of {len(act):,}  ({100*is_ext.mean():.1f} %)")
print(f"  so the best achievable CNEC match rate is "
      f"{100*(1-is_ext.mean()):.1f} %")
cnec = act[~is_ext].copy()

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("1  EIC COLUMNS AVAILABLE IN EACH FILE")
print("=" * 88)
d_eic = [c for c in dom.columns if "eic" in c]
a_eic = [c for c in act.columns if "eic" in c]
print(f"  domain: {d_eic}")
print(f"  active: {a_eic}")
for c in d_eic[:4]:
    print(f"    domain {c:<20} {dom[c].nunique():>6,} distinct, "
          f"{100*dom[c].notna().mean():>5.1f}% populated")
for c in a_eic[:4]:
    print(f"    active {c:<20} {cnec[c].nunique():>6,} distinct, "
          f"{100*cnec[c].notna().mean():>5.1f}% populated")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("2  CANDIDATE KEYS, SCORED")
print("=" * 88)

def rate(dparts, aparts, label):
    try:
        dk = dom["t"].astype("int64").astype(str)
        ak = cnec["t"].astype("int64").astype(str)
        for d, a in zip(dparts, aparts):
            dk = dk + "|" + d
            ak = ak + "|" + a
        hit = ak.isin(set(dk))
        dup = len(dom) / max(dk.nunique(), 1)
        print(f"  {label:<52} {100*hit.mean():>6.2f} %   "
              f"{dup:>4.2f} domain rows per key")
        return hit.mean()
    except Exception as exc:                                  # noqa: BLE001
        print(f"  {label:<52} failed: {str(exc)[:40]}")
        return 0.0


best, best_label = 0.0, None
cands = []

if "cneeic" in dom.columns:
    for a in a_eic:
        cands.append(([dom["cneeic"].astype(str)], [cnec[a].astype(str)],
                      f"element EIC:  domain.cneeic = active.{a}"))
        cands.append(([dom["cneeic"].astype(str),
                       dom["direction"].astype(str)],
                      [cnec[a].astype(str), cnec["direction"].astype(str)],
                      f"element EIC + direction:  cneeic = {a}"))

cands.append(([tidy(dom["cnename"])], [tidy(cnec[namecol])],
              "element NAME, tidied"))
cands.append(([tidy(dom["cnename"]), dom["direction"].astype(str)],
              [tidy(cnec[namecol]), cnec["direction"].astype(str)],
              "element NAME tidied + direction"))
cands.append(([tidy(dom["cnename"]), tidy(dom["contname"])],
              [tidy(cnec[namecol]), tidy(cnec["contname"])],
              "element + contingency NAME, both tidied"))

for dparts, aparts, label in cands:
    r = rate(dparts, aparts, label)
    if r > best:
        best, best_label = r, label

print(f"\n  best: {best_label}  at {100*best:.2f} %")
print("  'domain rows per key' near 1.00 means the key is unique - a key that")
print("  matches everything but is not unique cannot be used for the PTDFs.")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("3  PTDFs, COMPARED ON THE BEST UNIQUE KEY")
print("=" * 88)
ZONES = ["AT", "BE", "CZ", "DE", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK"]
dom["kk"] = (dom["t"].astype("int64").astype(str) + "|"
             + tidy(dom["cnename"]) + "|" + dom["direction"].astype(str))
cnec["kk"] = (cnec["t"].astype("int64").astype(str) + "|"
              + tidy(cnec[namecol]) + "|" + cnec["direction"].astype(str))
dd = dom.drop_duplicates("kk").set_index("kk")
m = cnec[cnec["kk"].isin(dd.index)]
print(f"  comparing {len(m):,} rows")
worst = 0.0
for z in ZONES:
    dcol, acol = f"ptdf_{z.lower()}", f"hub_{z.lower()}"
    if dcol in dd.columns and acol in m.columns:
        a = pd.to_numeric(m[acol], errors="coerce").values
        b = pd.to_numeric(dd.loc[m["kk"], dcol], errors="coerce").values
        d = pd.Series(abs(a - b)).dropna()
        if len(d):
            worst = max(worst, d.max())
            print(f"    {z}: max |diff| {d.max():.6f}   mean {d.mean():.6f}")
print(f"\n  worst disagreement: {worst:.6f}")
print("  below ~1e-4 means the domain's PTDFs and the published active ones")
print("  are the same numbers in the same convention, and the preprocessing")
print("  can use the domain as the model's constraint input.")
