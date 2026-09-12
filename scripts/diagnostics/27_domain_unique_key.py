"""Pin down a UNIQUE CNEC key, and settle the direction convention.

    python scripts/27_domain_unique_key.py

Script 26 got the match to 100% on (hour, element EIC) but that key is not
unique - 1.79 domain rows share it - because one element carries several
CNECs, one per contingency.  A non-unique key cannot be used to read PTDFs.

Two things to settle here:

  1. THE CONTINGENCY.  The domain names the contingency's branches in
     contBranchEic1..6.  activeFbConstraints has a single branchEic, which
     matched the element EIC only 33% of the time - consistent with it being
     the CONTINGENCY branch rather than the element.  If so,
     (hour, cneEic, contBranchEic1) is the unique CNEC identifier.

  2. DIRECTION.  Adding `direction` to a key that already matched 100% drops
     it to 81%, so the two files disagree on direction for a fifth of rows.
     OPPOSITE is the same wire with every PTDF sign flipped, so there is a
     sharp test: for the disagreeing rows, is |a - b| large while |a + b| is
     ~0?  If yes the convention differs by a sign and the handling is one
     line.  If neither is small, something else is wrong.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/raw/jao")
DOM  = ROOT / "domain2024"
ZONES = ["AT", "BE", "CZ", "DE", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK"]


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
namecol = "cnecname" if "cnecname" in act.columns else "cnename"
act = act[~(act[namecol].isna()
            | act[namecol].astype(str).str.contains("external", case=False,
                                                    na=False))]
print(f"domain {len(dom):,} rows   CNEC-only active {len(act):,} rows")

H = lambda df: df["t"].astype("int64").astype(str)                # noqa: E731

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("1  IS active.branchEic THE CONTINGENCY BRANCH?")
print("=" * 88)
for n in range(1, 7):
    c = f"contbrancheic{n}"
    if c not in dom.columns:
        continue
    dk = H(dom) + "|" + dom["cneeic"].astype(str) + "|" + dom[c].astype(str)
    ak = H(act) + "|" + act["cneceic"].astype(str) + "|" + \
        act["brancheic"].astype(str)
    hit = ak.isin(set(dk))
    print(f"  cneEic + {c:<16} match {100*hit.mean():>6.2f} %   "
          f"{len(dom)/max(dk.nunique(),1):>4.2f} domain rows per key")

# any of the six contingency branches
anyk = set()
for n in range(1, 7):
    c = f"contbrancheic{n}"
    if c in dom.columns:
        anyk |= set(H(dom) + "|" + dom["cneeic"].astype(str) + "|"
                    + dom[c].astype(str))
ak = H(act) + "|" + act["cneceic"].astype(str) + "|" + \
    act["brancheic"].astype(str)
print(f"\n  cneEic + ANY of contBranchEic1..6   match "
      f"{100*ak.isin(anyk).mean():>6.2f} %")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("2  UNIQUENESS OF THE CANDIDATE KEYS")
print("=" * 88)
keys = {
    "hour + cneEic":                  H(dom) + "|" + dom["cneeic"].astype(str),
    "hour + cneEic + direction":      H(dom) + "|" + dom["cneeic"].astype(str)
                                      + "|" + dom["direction"].astype(str),
    "hour + cneEic + contBranchEic1": H(dom) + "|" + dom["cneeic"].astype(str)
                                      + "|" + dom["contbrancheic1"].astype(str),
    "hour + cneEic + contBranchEic1 + direction":
        H(dom) + "|" + dom["cneeic"].astype(str) + "|"
        + dom["contbrancheic1"].astype(str) + "|"
        + dom["direction"].astype(str),
}
for label, k in keys.items():
    print(f"  {label:<46} {len(dom)/k.nunique():>5.2f} rows per key  "
          f"({k.nunique():,} keys)")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("3  DIRECTION: IS THE DISAGREEMENT A SIGN CONVENTION?")
print("=" * 88)
dom["ke"] = H(dom) + "|" + dom["cneeic"].astype(str) + "|" + \
    dom["contbrancheic1"].astype(str)
act["ke"] = H(act) + "|" + act["cneceic"].astype(str) + "|" + \
    act["brancheic"].astype(str)

d1 = dom.drop_duplicates("ke").set_index("ke")
m = act[act["ke"].isin(d1.index)].copy()
print(f"  rows joined on hour + element + contingency: {len(m):,}")
if len(m):
    m["dom_dir"] = d1.loc[m["ke"], "direction"].values
    print("\n  active direction vs domain direction:")
    print(pd.crosstab(m["direction"], m["dom_dir"]).to_string())

    same = m["direction"].astype(str).str.upper() == \
        m["dom_dir"].astype(str).str.upper()
    for label, sel in [("direction AGREES", same),
                       ("direction DIFFERS", ~same)]:
        if not sel.any():
            continue
        diffs, sums = [], []
        for z in ZONES:
            a = pd.to_numeric(m.loc[sel, f"hub_{z.lower()}"],
                              errors="coerce").values
            b = pd.to_numeric(d1.loc[m.loc[sel, "ke"], f"ptdf_{z.lower()}"],
                              errors="coerce").values
            diffs.append(np.abs(a - b))
            sums.append(np.abs(a + b))
        dv = np.nanmax(np.concatenate(diffs))
        sv = np.nanmax(np.concatenate(sums))
        dm = np.nanmean(np.concatenate(diffs))
        sm = np.nanmean(np.concatenate(sums))
        print(f"\n  {label}  ({int(sel.sum()):,} rows)")
        print(f"    |published - domain|   mean {dm:.6f}   max {dv:.6f}")
        print(f"    |published + domain|   mean {sm:.6f}   max {sv:.6f}")
        verdict = ("SAME sign" if dm < sm else "OPPOSITE sign")
        print(f"    -> the two files use the {verdict} convention here")

print("\n  If the AGREES block has a near-zero difference and the DIFFERS")
print("  block a near-zero SUM, the whole thing is one sign flip keyed on")
print("  direction, and the preprocessing multiplies OPPOSITE rows by -1.")
