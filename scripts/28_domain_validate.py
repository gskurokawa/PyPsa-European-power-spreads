"""Final validation: on an unambiguous key, does the domain match reality?

    python scripts/28_domain_validate.py

Script 27 established two things.  The join works on machine identifiers -
(hour, cneEic, contBranchEic1) matches 100% of CNEC rows.  And where that key
lands on the right domain row the PTDFs agree to a mean of 0.000188, which is
exact for this purpose.  What it could not do is guarantee it landed on the
right row: 1.12 to 1.33 domain rows share the key, so a third of comparisons
were against a NEIGHBOURING CNEC - same element, different contingency - whose
PTDFs differ by about 0.02.  That, not a sign convention, is what the earlier
disagreement was.

So this compares only on keys that identify exactly ONE domain row, where
there is nothing to get wrong, and checks two independent quantities:

  PTDF  - should agree to ~1e-6
  RAM   - published in BOTH files, so it is a second, independent check that
          the row we matched really is the row that bound

A note on why full uniqueness is not required for the BUILD.  The join to
activeFbConstraints is a validation device, not part of the model.  The model
consumes the domain directly: every domain row IS a constraint, and two rows
sharing a name are two constraints, which is correct rather than a problem.
Uniqueness only matters here, for checking.
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
nc = "cnecname" if "cnecname" in act.columns else "cnename"
act = act[~(act[nc].isna()
            | act[nc].astype(str).str.contains("external", case=False,
                                               na=False))]

H = lambda df: df["t"].astype("int64").astype(str)                # noqa: E731

# progressively tighter keys; use the tightest that still matches everything
CANDIDATES = [
    ("hour+element+cont1+direction+status",
     lambda d, c1, c2: (H(d) + "|" + d["cneeic"].astype(str) + "|"
                        + d[c1].astype(str) + "|" + d["direction"].astype(str)
                        + "|" + d.get("cnestatus", "").astype(str))),
    ("hour+element+cont1+direction",
     lambda d, c1, c2: (H(d) + "|" + d["cneeic"].astype(str) + "|"
                        + d[c1].astype(str) + "|"
                        + d["direction"].astype(str))),
    ("hour+element+cont1",
     lambda d, c1, c2: (H(d) + "|" + d["cneeic"].astype(str) + "|"
                        + d[c1].astype(str))),
]

print("=" * 88)
print("KEYS, AND HOW MUCH OF THE DOMAIN THEY DISAMBIGUATE")
print("=" * 88)
for label, fn in CANDIDATES:
    k = fn(dom, "contbrancheic1", None)
    n = k.value_counts()
    print(f"  {label:<40} {len(dom)/k.nunique():>5.2f} rows/key   "
          f"{100*n.eq(1).mean():>5.1f} % of keys are unique")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("COMPARED ONLY WHERE THE KEY IS UNAMBIGUOUS")
print("=" * 88)
dom["k"] = (H(dom) + "|" + dom["cneeic"].astype(str) + "|"
            + dom["contbrancheic1"].astype(str) + "|"
            + dom["direction"].astype(str))
act["k"] = (H(act) + "|" + act["cneceic"].astype(str) + "|"
            + act["brancheic"].astype(str) + "|"
            + act["direction"].astype(str))

counts = dom["k"].value_counts()
uniq = set(counts[counts == 1].index)
d1 = dom[dom["k"].isin(uniq)].set_index("k")
m = act[act["k"].isin(uniq)].copy()
print(f"  active CNEC rows            : {len(act):,}")
print(f"  of those, on an unambiguous key: {len(m):,} "
      f"({100*len(m)/max(len(act),1):.1f} %)")

if len(m):
    print("\n  PTDF, published vs domain:")
    alld = []
    for z in ZONES:
        a = pd.to_numeric(m[f"hub_{z.lower()}"], errors="coerce").values
        b = pd.to_numeric(d1.loc[m["k"], f"ptdf_{z.lower()}"],
                          errors="coerce").values
        d = np.abs(a - b)
        alld.append(d)
        print(f"    {z}:  mean {np.nanmean(d):.8f}   max {np.nanmax(d):.6f}"
              f"   >1e-4 in {int(np.nansum(d > 1e-4)):,} of {len(d):,}")
    alld = np.concatenate(alld)
    print(f"\n  overall: mean {np.nanmean(alld):.8f}   "
          f"max {np.nanmax(alld):.6f}   "
          f"{100*np.nanmean(alld < 1e-4):.2f} % within 1e-4")

    if "ram" in m.columns and "ram" in d1.columns:
        a = pd.to_numeric(m["ram"], errors="coerce").values
        b = pd.to_numeric(d1.loc[m["k"], "ram"], errors="coerce").values
        r = np.abs(a - b)
        print(f"\n  RAM, published vs domain (independent second check):")
        print(f"    mean |diff| {np.nanmean(r):.4f} MW   "
              f"max {np.nanmax(r):.1f} MW   "
              f"{100*np.nanmean(r < 1.0):.1f} % within 1 MW")

print("\n" + "=" * 88)
print("VERDICT")
print("=" * 88)
print("  PTDF within 1e-4 and RAM within 1 MW on the unambiguous rows means")
print("  the domain parse, the hub mapping and the hour alignment are all")
print("  correct, and the domain can be used as the model's constraint input.")
print("  The remaining ambiguity is between neighbouring CNECs on the same")
print("  element and does not affect the build, which consumes every domain")
print("  row as its own constraint.")
