"""Which RAM definition, and which net position, make the constraint true?

    python scripts/32_ram_reference.py

Script 31 got the arithmetic running and found real signal - a net position
summed over ALL borders halves the binding-constraint margin and doubles the
separation from slack ones - but binding constraints still sit about 279 MW
from zero instead of on it.  A constant offset of that size, against an
average RAM of 882 MW, points at the REFERENCE FLOW.

THE POINT
---------
"Remaining available margin" is a margin measured from somewhere, and the
domain publishes several candidates for that somewhere:

    fmax        thermal rating of the element
    frm         reliability margin
    f0core      flow when all CORE net positions are zero
    f0all       flow when ALL net positions are zero
    fref        flow expected in the reference case, which has its OWN
                non-zero net positions
    frefinit    the same before adjustments

The constraint is  SUM PTDF x NP <= RAM  only when RAM is measured from the
ZERO-net-position reference.  If the published RAM is measured from fref
instead, the true constraint is  SUM PTDF x (NP - NP_ref) <= RAM, and using
absolute net positions leaves precisely a constant offset - which is what
script 31 shows.

So rather than argue about JAO's methodology, this builds every plausible RAM
and crosses it with every plausible net position, and reports which cell puts
the binding constraints on zero.  Same approach that settled the sign
convention in section 12 and the join key in scripts 26-28.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW  = ROOT / "data" / "raw"
DOM  = RAW / "jao" / "domain2024"

MODEL_ZONES = ["DE_LU", "FR", "PL", "NL", "BE", "AT", "CH", "CZ"]
CORE_ZONES  = ["DE_LU", "FR", "PL", "NL", "BE", "AT", "CZ"]     # CH not in Core


def norm(df):
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿").lower() for c in df.columns]
    return df


# ------------------------------------------------------------------
print("=" * 88)
print("1  HOW THE PUBLISHED RAM RELATES TO ITS INGREDIENTS")
print("=" * 88)
frames = []
for f in sorted(DOM.glob("fc_*.csv.gz")):
    try:
        frames.append(pd.read_csv(f, low_memory=False))
    except Exception:                                         # noqa: BLE001
        pass
d = norm(pd.concat(frames, ignore_index=True))
d["t"] = pd.to_datetime(d["datetimeutc"], utc=True, errors="coerce")
d["cid"] = (d["cneeic"].astype(str) + "|" + d["contbrancheic1"].astype(str)
            + "|" + d["direction"].astype(str))
d = d.drop_duplicates(subset=["t", "cid"])

INGREDIENTS = ["fmax", "frm", "fref", "frefinit", "f0core", "f0all",
               "fuaf", "amr", "cva", "iva", "ltamargin", "fnrao", "ram"]
have = [c for c in INGREDIENTS if c in d.columns]
for c in have:
    d[c] = pd.to_numeric(d[c], errors="coerce")
print(d[have].describe().loc[["mean", "std", "min", "max"]].round(1).to_string())

print("\n  how close is each reconstruction to the published ram?")
recon = {
    "fmax - frm - fref":     lambda x: x["fmax"] - x["frm"] - x["fref"],
    "fmax - frm - f0core":   lambda x: x["fmax"] - x["frm"] - x["f0core"],
    "fmax - frm - f0all":    lambda x: x["fmax"] - x["frm"] - x["f0all"],
    "fmax - frm - frefinit": lambda x: x["fmax"] - x["frm"] - x["frefinit"],
}
for label, fn in recon.items():
    try:
        r = fn(d)
        err = (r - d["ram"]).abs()
        print(f"    {label:<26} mean |diff| {err.mean():>8.1f} MW   "
              f"within 1 MW: {100*(err < 1).mean():>5.1f} %")
    except Exception as exc:                                  # noqa: BLE001
        print(f"    {label:<26} n/a ({str(exc)[:40]})")

# ------------------------------------------------------------------
print("\n" + "=" * 88)
print("2  THE GRID: RAM DEFINITION x NET POSITION DEFINITION")
print("=" * 88)

dom = pd.read_parquet(PROC / "fb_domain.parquet")
dom["t"] = pd.to_datetime(dom["t"], utc=True)
dom = dom.merge(d[["t", "cid"] + [c for c in have if c != "ram"]],
                on=["t", "cid"], how="left")

om = [c for c in dom.columns if c.startswith("fixed_")]
dom["omitted_mw"] = dom[om].abs().sum(axis=1) * 4000
sub = dom[dom["omitted_mw"] < 50].copy()

act = norm(pd.read_csv(RAW / "jao" / "activeFbConstraints_2024.csv",
                       low_memory=False))
act["t"] = pd.to_datetime(act["datetimeutc"], utc=True, errors="coerce")
act = act[act["shadowprice"].abs() > 1e-9]
act["cid"] = (act["cneceic"].astype(str) + "|" + act["brancheic"].astype(str)
              + "|" + act["direction"].astype(str))
bound = set(act["t"].astype("int64").astype(str) + "|" + act["cid"])
sub["bound"] = (sub["t"].astype("int64").astype(str) + "|"
                + sub["cid"]).isin(bound)
print(f"  judging {len(sub):,} constraints, of which "
      f"{int(sub['bound'].sum()):,} bound")

idx = pd.DatetimeIndex(sub["t"])


def np_from_flows(path, zones):
    nf = pd.read_parquet(PROC / path)
    if nf.index.tz is None:
        nf.index = nf.index.tz_localize("UTC")
    out = pd.DataFrame(index=nf.index)
    for z in zones:
        tot = pd.Series(0.0, index=nf.index)
        for c in nf.columns:
            if ">" not in str(c):
                continue
            a, b = str(c).split(">")
            if a == z:
                tot = tot + nf[c].fillna(0.0)
            elif b == z:
                tot = tot - nf[c].fillna(0.0)
        out[z] = tot
    return out


bp = pd.read_parquet(PROC / "boundary_position.parquet")
if bp.index.tz is None:
    bp.index = bp.index.tz_localize("UTC")

NPDEFS = {
    "boundary (13 modelled borders)": bp,
    "total (all borders)":            np_from_flows("net_flows.parquet",
                                                    MODEL_ZONES),
    "Core borders only (no CH)":      np_from_flows("commercial_flows.parquet",
                                                    CORE_ZONES),
}
RAMDEFS = {"published ram": lambda x: x["ram"]}
for label, fn in recon.items():
    RAMDEFS[label] = fn

print(f"\n  {'RAM':<26}{'net position':<32}{'bound med':>11}"
      f"{'slack med':>11}{'<50MW':>8}{'sep':>8}")
best = None
for rlabel, rfn in RAMDEFS.items():
    try:
        ram = rfn(sub).values
    except Exception:                                         # noqa: BLE001
        continue
    if not np.isfinite(ram).any():
        continue
    for nlabel, npdf in NPDEFS.items():
        npx = npdf.reindex(idx)
        lhs = np.zeros(len(sub))
        for z in MODEL_ZONES:
            col = f"ptdf_{z}"
            if col in sub.columns and z in npx.columns:
                lhs = lhs + sub[col].values * np.nan_to_num(npx[z].values)
        margin = ram - lhs
        b = margin[sub["bound"].values]
        s = margin[~sub["bound"].values]
        b, s = b[np.isfinite(b)], s[np.isfinite(s)]
        if not len(b):
            continue
        hit = 100 * np.mean(np.abs(b) < 50)
        sep = np.median(s) - np.median(b)
        print(f"  {rlabel:<26}{nlabel:<32}{np.median(b):>11.0f}"
              f"{np.median(s):>11.0f}{hit:>7.0f}%{sep:>8.0f}")
        if best is None or abs(np.median(b)) < abs(best[0]):
            best = (np.median(b), rlabel, nlabel, hit, sep)

if best:
    print(f"\n  closest to zero: {best[1]}  x  {best[2]}")
    print(f"    binding median {best[0]:.0f} MW, {best[3]:.0f}% within 50 MW, "
          f"separation {best[4]:.0f} MW")
print("\n  A binding median near zero identifies the right pair.  If every")
print("  cell is far from zero, the missing piece is not the reference flow")
print("  and the next suspect is the net position CONVENTION - whether JAO")
print("  measures it against a reference case rather than against zero.")


# ==================================================================
# 3  JAO'S OWN NET POSITIONS - the definition the constraints use
# ==================================================================
# Everything above builds Core net positions from this repo's thirteen
# modelled borders, which is incomplete for Austria, Czechia and Poland (their
# borders with Hungary, Slovakia, Slovenia and Croatia are missing) and absent
# for the five Core zones the model does not represent.  That is why the best
# cell reached only a 22% hit rate.
#
# JAO publishes netPos: the hourly CORE net position of every Core zone, by
# the same body that publishes the constraints.  There is no definitional gap
# between the two, which is exactly what an ENTSO-E series could not promise.
#
# Sign is NOT assumed.  Germany reads +10,202 and France -4,684 for one hour,
# which is the opposite of what "positive = export" would suggest for 2024, so
# both signs are tested and the data decides - the same approach that settled
# the leading minus in section 12 of FINDINGS.md.
LOG = open(ROOT / "logs" / "ram_reference.txt", "w", encoding="utf-8")


def out2(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    LOG.write(line + "\n")


out2("\n" + "=" * 96)
out2("3  JAO netPos - ALL TWELVE CORE ZONES, BOTH SIGNS")
out2("=" * 96)

npf = RAW / "jao" / "netPos_2024.csv"
if not npf.exists():
    out2("  netPos_2024.csv not present - run script 22 --phase1-only first")
else:
    jn = norm(pd.read_csv(npf, low_memory=False))
    jn["t"] = pd.to_datetime(jn["datetimeutc"], utc=True, errors="coerce")
    jn = jn.set_index("t")
    jn = jn[~jn.index.duplicated(keep="first")]
    out2(f"  netPos: {len(jn):,} hours, "
         f"{jn.index.min()} .. {jn.index.max()}")

    # fb_domain columns -> the netPos column carrying that zone
    COL = {"ptdf_DE_LU": "hub_de", "ptdf_FR": "hub_fr", "ptdf_PL": "hub_pl",
           "ptdf_NL": "hub_nl", "ptdf_BE": "hub_be", "ptdf_AT": "hub_at",
           "ptdf_CZ": "hub_cz",
           "fixed_HR": "hub_hr", "fixed_HU": "hub_hu", "fixed_RO": "hub_ro",
           "fixed_SI": "hub_si", "fixed_SK": "hub_sk"}
    have = {k: v for k, v in COL.items()
            if k in sub.columns and v in jn.columns}
    out2(f"  matched {len(have)} of 12 Core zones: "
         f"{', '.join(sorted(k.split('_', 1)[1] for k in have))}")
    missing = [k for k in COL if k not in have]
    if missing:
        out2(f"  NOT matched: {missing}")

    npx = jn.reindex(idx)
    cover = 100 * npx[[v for v in have.values()]].notna().all(axis=1).mean()
    out2(f"  hours with a full set of net positions: {cover:.1f} %")

    out2(f"\n  {'RAM':<26}{'sign':>6}{'CH':>5}{'bound med':>11}"
         f"{'slack med':>11}{'<50MW':>8}{'<100MW':>9}{'sep':>8}")
    best2 = None
    for rlabel, rfn in RAMDEFS.items():
        try:
            ram = rfn(sub).values
        except Exception:                                     # noqa: BLE001
            continue
        if not np.isfinite(ram).any():
            continue
        base = np.zeros(len(sub))
        for pcol, ncol in have.items():
            base = base + sub[pcol].values * np.nan_to_num(npx[ncol].values)
        # Switzerland is not a Core zone so JAO publishes no netPos for it,
        # but the domain does carry a ptdf_ch. Test with it left out and with
        # this repo's own boundary figure standing in.
        ch = np.zeros(len(sub))
        if "ptdf_CH" in sub.columns:
            chn = bp.reindex(idx)["CH"] if "CH" in bp.columns else None
            if chn is not None:
                ch = sub["ptdf_CH"].values * np.nan_to_num(chn.values)
        for sgn in (+1, -1):
            for ch_label, ch_term in (("no", np.zeros(len(sub))), ("yes", ch)):
                lhs = sgn * (base + ch_term)
                margin = ram - lhs
                b = margin[sub["bound"].values]
                s_ = margin[~sub["bound"].values]
                b, s_ = b[np.isfinite(b)], s_[np.isfinite(s_)]
                if not len(b):
                    continue
                h50 = 100 * np.mean(np.abs(b) < 50)
                h100 = 100 * np.mean(np.abs(b) < 100)
                sep = np.median(s_) - np.median(b)
                out2(f"  {rlabel:<26}{'+' if sgn > 0 else '-':>6}"
                     f"{ch_label:>5}{np.median(b):>11.0f}{np.median(s_):>11.0f}"
                     f"{h50:>7.0f}%{h100:>8.0f}%{sep:>8.0f}")
                if best2 is None or h50 > best2[0]:
                    best2 = (h50, rlabel, sgn, ch_label,
                             np.median(b), sep, h100)
    if best2:
        out2(f"\n  BEST: {best2[1]}, sign {'+' if best2[2] > 0 else '-'}, "
             f"CH {best2[3]}")
        out2(f"    binding median {best2[4]:.0f} MW, "
             f"{best2[0]:.0f}% within 50 MW, {best2[6]:.0f}% within 100 MW, "
             f"separation {best2[5]:.0f} MW")
    out2("\n  A high share within 50 MW plus a clear positive separation means")
    out2("  the pipeline reproduces the real constraint set and the CNEC build")
    out2("  can proceed. This supersedes section 2, which used incomplete net")
    out2("  positions and could only reach 22%.")

LOG.close()
print("\nwritten to logs/ram_reference.txt")
