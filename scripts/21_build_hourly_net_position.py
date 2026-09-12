"""Turn the JAO CORE domain into hourly net-position bounds the model can use.

    .venv-data\\Scripts\\activate          <- pandas + pyarrow, no pypsa needed
    python scripts/21_build_hourly_net_position.py
    python scripts/21_build_hourly_net_position.py --mode raw

Then DEACTIVATE and go back to .venv before running the model: 20 and 21 are
data steps and live in .venv-data, 10_build_network.py needs pypsa and lives in
.venv. See requirements-data.txt for why the two are separate.

Reads data/raw/jao/maxNetPos_*.json, writes
data/processed/net_position_hourly.parquet with columns "<zone>|hi" and
"<zone>|lo".

The scope mismatch, stated first because it decides the method
------------------------------------------------------------
JAO's max/min net position is a hub's total position in the CORE coupling,
across every CORE border. The model's net position is the sum over its own
eight zones' internal links only. For Germany those differ by the Danish,
Swedish and Swiss borders; for Poland by Sweden and Lithuania; and CORE
includes Hungary, Romania, Slovakia, Slovenia and Croatia, which the model does
not represent at all. The two numbers are not the same quantity, and using
JAO's directly would repeat the error that the commercial-versus-physical
net-position test just made - bounding one quantity with the envelope of
another.

So the default is --mode shape:

    hourly_bound(zone, t) = static_bound(zone) * jao(zone, t) / median(jao(zone))

LEVEL from the model's own scope, where the observed-flow quantile in
12_build_net_position.py is the right calibration; SHAPE from the published
domain, which is the hour-to-hour movement the model has no other way to know
and the only thing that can put its congestion in the right HOURS.

The ratio is clipped, because a domain that collapses to near zero in one hour
is a real event for the whole CORE region but not something a bound derived
from a different scope should inherit at full force.

--mode raw uses JAO's numbers as they stand. It is wrong on scope and is here
so the difference can be measured rather than argued about.

Switzerland is not in CORE and keeps its static bound. So does any zone or hour
JAO does not cover: the model falls back to the constant, never to nothing.

RESULT: REFUTED (2024 hold-out at 100% JAO coverage, run 2026-09-05)
--------------------------------------------------------------------
The input was good and the answer is no.

  shape_sd 0.11-0.33 across all seven CORE zones - the published domain really
  does move 11-33% hour to hour where the model had one constant.

  It changed the model a lot: DE-FR spread sd 0.90x -> 1.50x, DE-PL 1.03x ->
  1.61x. Tighter bounds in some hours congest harder, exactly as expected.

  It changed the TIMING not at all. hit_% still equals if_random_% on all
  thirteen borders (DE-PL 75.2 vs 70.9, DE-FR 55.7 vs 52.2), phi moved by
  0.01-0.02, and both hourly correlations got WORSE - DE-FR 0.43 -> 0.30,
  DE-PL 0.29 -> 0.23. The extra volatility landed in the wrong hours.

  7/15 -> 5/15, losing both spread-sd criteria.

WHY, and it is not a bug. max net position for a zone is the maximum of that
zone's position over the WHOLE polytope - attained at a vertex where every
other zone is arranged to make this one's export extreme. It is an unrepresen-
tative corner, and it widens or narrows for reasons largely unrelated to
whether that zone was actually constrained in that hour, or in the direction it
wanted to trade. The projections carry the domain's SIZE. The timing is in its
SHAPE, and projecting a polytope onto its axes is precisely the operation that
discards shape.

Damping the multiplier cannot rescue it. The timing effect is null at full
strength, so a milder version only walks back toward the constant bound; it
would recover the two lost criteria and gain nothing.

What would work is the CNEC constraints themselves - sum of PTDF(z,c) * NP(z)
<= RAM(c), from the finalComputation endpoint - which is the actual polytope
rather than its shadow. Costed in the conversation of 2026-09-05 and not taken:
see the note in validate.price_separation().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402

from spread.config import PROCESSED, load_config             # noqa: E402

RAW = ROOT / "data" / "raw" / "jao"

# CORE publishes DE and LU as one hub, written "DE" or "DE_LU" depending on the
# endpoint. Everything else matches the model's code.
HUB_TO_ZONE = {"DE": "DE_LU", "DE_LU": "DE_LU", "ALDE": None, "ALBE": None}


def _zone(hub: str, zones: list[str]) -> str | None:
    hub = hub.strip().upper()
    if hub in HUB_TO_ZONE:
        return HUB_TO_ZONE[hub]
    return hub if hub in zones else None


def tidy(rows: list, zones: list[str]) -> pd.DataFrame:
    """Normalise JAO rows to a frame of '<zone>|hi' / '<zone>|lo' columns.

    The endpoint's exact shape is not documented, so both plausible ones are
    handled: long (a row per hub, with a hub column) and wide (a row per hour,
    with a column per hub). Run 20_fetch_jao_domain.py --probe to see which.
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    tcol = next((c for c in frame.columns
                 if c.lower() in ("datetimeutc", "dateutc", "timestamputc",
                                  "datetime", "mtu", "date")), None)
    if tcol is None:
        raise SystemExit(f"no time column found in {list(frame.columns)[:12]}")
    frame[tcol] = pd.to_datetime(frame[tcol], utc=True, errors="coerce")
    frame = frame.dropna(subset=[tcol])

    hubcol = next((c for c in frame.columns
                   if c.lower() in ("hub", "biddingzone", "zone", "area")), None)

    out: dict[str, pd.Series] = {}
    if hubcol is not None:                                   # long
        maxc = next((c for c in frame.columns if "max" in c.lower()), None)
        minc = next((c for c in frame.columns if "min" in c.lower()), None)
        if maxc is None or minc is None:
            raise SystemExit(f"no min/max columns in {list(frame.columns)[:12]}")
        for hub, part in frame.groupby(hubcol):
            zone = _zone(str(hub), zones)
            if zone is None:
                continue
            part = part.set_index(tcol).sort_index()
            out[f"{zone}|hi"] = pd.to_numeric(part[maxc], errors="coerce")
            out[f"{zone}|lo"] = pd.to_numeric(part[minc], errors="coerce")
    else:                                                    # wide
        frame = frame.set_index(tcol).sort_index()
        for col in frame.columns:
            low = col.lower()
            if "max" in low:
                side = "hi"
            elif "min" in low:
                side = "lo"
            else:
                continue
            # Strip the prefix to leave the hub code. Keep underscores: the
            # real schema has maxDE alongside maxDE_DK1_VH and maxPL_LT_BigHub,
            # which are hybrid-interconnector hubs, not zones, and must not
            # collapse onto DE and PL.
            hub = col
            for prefix in ("maxNetPos", "minNetPos", "max", "min"):
                if hub.lower().startswith(prefix.lower()):
                    hub = hub[len(prefix):]
                    break
            zone = _zone(hub, zones)
            if zone is None:
                continue
            out[f"{zone}|{side}"] = pd.to_numeric(frame[col], errors="coerce")
    if not out:
        raise SystemExit(f"no hubs matched the model's zones. columns seen: "
                         f"{list(frame.columns)[:20]}")
    return pd.DataFrame(out).sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("shape", "raw"), default="shape")
    ap.add_argument("--min-rows", type=int, default=2000,
                    help="skip a JAO file with fewer rows than this")
    ap.add_argument("--clip", type=float, nargs=2, default=(0.35, 1.75),
                    help="bounds on the shape ratio, as a multiple of median")
    args = ap.parse_args()

    zones = load_config()["zones"]
    files = sorted(RAW.glob("maxNetPos_*.json"))
    if not files:
        raise SystemExit(f"no JAO files in {RAW} - run scripts/20_fetch_jao_domain.py")

    # A stale or half-finished file must not be averaged in: the shape ratio
    # is taken against each series' MEDIAN, so a file holding one December day
    # would drag the whole year's multiplier toward that day.
    rows: list = []
    for f in files:
        part = json.loads(f.read_text(encoding="utf-8"))
        if len(part) < args.min_rows:
            print(f"  SKIPPING {f.name}: {len(part)} rows, below --min-rows "
                  f"{args.min_rows}. Refetch it or delete it.")
            continue
        print(f"  {f.name}: {len(part):,} rows")
        rows.extend(part)
    if not rows:
        raise SystemExit("every JAO file was skipped - nothing to build from")
    print(f"  {len(rows):,} rows total")

    jao = tidy(rows, zones)
    jao = jao[~jao.index.duplicated(keep="last")]
    print(f"  {len(jao):,} hours, {jao.shape[1]} series, "
          f"{jao.index.min()} -> {jao.index.max()}")

    static = pd.read_csv(PROCESSED / "net_position.csv", index_col=0)
    lo_ratio, hi_ratio = args.clip

    out = {}
    report = []
    for zone in zones:
        for side, col in (("hi", "hi_mw"), ("lo", "lo_mw")):
            key = f"{zone}|{side}"
            if key not in jao or zone not in static.index:
                continue
            s = jao[key].astype(float)
            if args.mode == "raw":
                built = s
                ratio_sd = float("nan")
            else:
                med = float(s.abs().median())
                if not med:
                    continue
                ratio = (s.abs() / med).clip(lo_ratio, hi_ratio)
                built = float(static.at[zone, col]) * ratio
                ratio_sd = float(ratio.std())
            out[key] = built
            report.append({
                "zone": zone, "side": side,
                "static_mw": round(float(static.at[zone, col])),
                "mean_mw": round(float(built.mean())),
                "p05_mw": round(float(built.quantile(0.05))),
                "p95_mw": round(float(built.quantile(0.95))),
                "shape_sd": round(ratio_sd, 3),
                "hours": int(built.notna().sum()),
            })

    if not out:
        raise SystemExit("nothing built - check the schema with --probe")
    frame = pd.DataFrame(out).sort_index()
    frame.to_parquet(PROCESSED / "net_position_hourly.parquet")

    print("\n" + "=" * 92)
    print(f"HOURLY NET POSITION BOUNDS   mode={args.mode}")
    print("=" * 92)
    print(pd.DataFrame(report).set_index(["zone", "side"]).to_string())
    missing = [z for z in zones if f"{z}|hi" not in frame]
    if missing:
        print(f"\n  not in CORE, keeping their static bound: {', '.join(missing)}")
    print("\n  shape_sd is the standard deviation of the hourly multiplier. Near")
    print("  zero means the published domain barely moves for that zone, and")
    print("  this whole exercise cannot help it. Large means the constant bound")
    print("  the model has been using was hiding real hourly variation.")
    print(f"\n  wrote {PROCESSED / 'net_position_hourly.parquet'}")


if __name__ == "__main__":
    main()
