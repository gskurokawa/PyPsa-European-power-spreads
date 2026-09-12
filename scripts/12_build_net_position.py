"""Derive each zone's net-position bounds from observed border flows.

    python scripts/12_build_net_position.py
    python scripts/12_build_net_position.py --quantile 0.995 --headroom 1.05

What this is
------------
Flow-based market coupling constrains the combination of exchanges, not each
border separately: every trade competes for room on the same physical wires.
A transport model has no such shared constraint, so it saturates all thirteen
borders at once and moves far more power than Europe really does.

The minimal stand-in is a bound on each zone's net position - the sum of its
exports minus its imports across all internal borders. Borders keep their full
thermal rating and stay uncongested in most hours, exactly as observed, but a
zone cannot export across all of them simultaneously.

The bound is taken from what each zone actually did: a high quantile of the
observed hourly net position, with a little headroom so the calibration period
is not reproduced exactly. Quantile rather than max, because a single hour of
unusual flow should not set a limit that stands for the whole study.

Writes data/processed/net_position.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402

from spread.config import PROCESSED, load_config             # noqa: E402
from spread.network import observed_net_position             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantile", type=float, default=0.995,
                    help="quantile of observed net position used as the bound")
    ap.add_argument("--headroom", type=float, default=1.05,
                    help="multiplier on the bound, so the model is not pinned "
                         "to the calibration period")
    ap.add_argument("--source", choices=("commercial", "wider"),
                    default="commercial",
                    help="commercial: bound the quantity the links carry, "
                         "falling back to the wider series only for zones "
                         "outside market coupling. wider: the old rule.")
    ap.add_argument("--year", type=int, default=None,
                    help="restrict to one year (default: all available)")
    args = ap.parse_args()

    cfg = load_config()
    zones = cfg["zones"]

    # A net-position bound is a statement about what a zone was ABLE to do.
    # Commercial exchange and physical flow are both observations of that, so
    # the bound is the wider of the two: taking the narrower one asserts a zone
    # could not do something it demonstrably did.
    #
    # This is not hedging. It matters most for Switzerland, which is outside EU
    # market coupling - excluded from SDAC since the framework-agreement
    # dispute - so its cross-border trade is bilateral and NTC-based rather
    # than auction-allocated, and much of what crosses its borders is transit.
    # Its scheduled commercial exchange understates what it actually does by
    # over a gigawatt, and the commercial-only bound made the model shed 9.9
    # GWh of Swiss load at VOLL in 35 hours.
    #
    # Links keep the commercial series alone, where the allocation argument is
    # exact - and where it recovered Alegro's 1,000 MW rating to the megawatt.
    sources = {}
    for label, name in (("commercial", "exchange.parquet"),
                        ("physical", "net_flows.parquet")):
        path = PROCESSED / name
        if not path.exists():
            continue
        flows = pd.read_parquet(path)
        if args.year:
            flows = flows[flows.index.year == args.year]
        sources[label] = observed_net_position(flows, zones)
    del label
    if not sources:
        raise SystemExit("no exchange series found - run scripts/02_build_hourly.py")
    print(f"  net positions from the wider of: {', '.join(sources)}\n")

    # Constrain the quantity the model actually moves.
    #
    # The links carry COMMERCIAL exchange (zones.yaml capacity_source), except
    # on borders touching a zone outside market coupling, where blend_exchange
    # falls back to physical flow. Bounding a commercial-flow model with a
    # physical-flow envelope charges a zone for transit it never traded: power
    # scheduled Germany-to-Austria that routes through Poland and Czechia
    # appears in their physical net position and in nobody's commercial one.
    #
    # The "wider of the two" rule above was written for a reason that is real
    # but narrow - Switzerland's commercial-only bound made the model shed
    # 9.9 GWh of Swiss load at VOLL in 35 hours - and it was then applied to
    # every zone. That is the same failure mode as the three separate Swiss
    # breakages: a decision that belongs to one zone, made globally.
    #
    # So: commercial where the model is commercial, wider where it is not.
    # --source wider restores the old behaviour for comparison.
    #
    # MEASURED (2024 hold-out, run 2026-09-05). The mechanism was real and the
    # prediction about WHERE was wrong. France was already bounded by its
    # commercial series, so its bounds did not move by a megawatt and the
    # predicted French deterioration never happened. The zones that moved were
    # the transit corridor - CZ import bound -3167 -> -2142 (32% tighter),
    # PL -2817 -> -2533, DE_LU -11777 -> -11152. Germany-to-Austria routing
    # through Czechia and Poland is exactly the transit the argument named;
    # France is not a transit country in this footprint.
    #
    # Still 7/15, and it cost headroom: DE-PL spread sd 1.03x -> 1.17x against
    # a 1.25x limit, DE-PL hourly correlation 0.29 -> 0.27, DE-PL mean spread
    # -2.53 -> -3.53. Tighter corridor bounds congest the corridor more, which
    # widens and destabilises the DE-PL spread.
    #
    # The counter-argument, which is not weak: a net-position bound is a
    # stand-in for the flow-based domain, and the domain is a PHYSICAL
    # constraint. A Czech zone whose wires carry German-to-Austrian transit
    # genuinely has less room to trade, and the physical envelope encodes that
    # while the commercial one discards it. Against that, the model's
    # constraint is written on commercial link flows, so a physical bound
    # compares two different quantities.
    #
    # Neither argument is decided by the criteria count, which is 7/15 either
    # way. What decides it is that this project's subject is the DISTRIBUTION
    # of spreads: a base case at 1.17x of a 1.25x tolerance will fail that
    # criterion in a large share of Monte Carlo draws, and an over-volatile
    # DE-PL spread is a substantive defect for the question being asked, not a
    # scoreboard problem.
    outside = set(cfg.get("outside_market_coupling", []))
    rows = []
    for zone in zones:
        parts = {k: v[zone].dropna() for k, v in sources.items()
                 if zone in v and not v[zone].dropna().empty}
        if not parts:
            continue
        his = {k: float(v.quantile(args.quantile)) for k, v in parts.items()}
        los = {k: float(v.quantile(1 - args.quantile)) for k, v in parts.items()}

        use_wider = (args.source == "wider" or zone in outside
                     or "commercial" not in parts)
        if use_wider:
            hi, lo = max(his.values()), min(los.values())
            picked = f"{min(los, key=los.get)[:4]}/{max(his, key=his.get)[:4]}"
        else:
            hi, lo = his["commercial"], los["commercial"]
            picked = "comm/comm"
        hi, lo = hi * args.headroom, lo * args.headroom

        s = parts.get("commercial", next(iter(parts.values())))
        row = {
            "zone": zone,
            "lo_mw": round(lo),
            "hi_mw": round(hi),
            "binds": picked,
            "obs_mean_mw": round(float(s.mean())),
            "hours": int(len(s)),
        }
        # Show what the old rule would have given, so the change is auditable
        # rather than silent.
        row["wider_lo"] = round(min(los.values()) * args.headroom)
        row["wider_hi"] = round(max(his.values()) * args.headroom)
        row["hi_cut_%"] = round(100 * (row["hi_mw"] / row["wider_hi"] - 1), 1) \
            if row["wider_hi"] else 0.0
        rows.append(row)

    out = pd.DataFrame(rows).set_index("zone")
    out[["lo_mw", "hi_mw"]].to_csv(PROCESSED / "net_position.csv")

    print("=" * 78)
    print("OBSERVED NET POSITION BY ZONE, MW   (positive = net exporter)")
    print("=" * 78)
    print(out.to_string())
    print(f"\n  bound = {args.quantile:.3%} quantile x {args.headroom:.2f} headroom,")
    print(f"  source = {args.source}. The 'binds' column names the series used")
    print("  for the lower and upper bound: comm = commercial, phys = physical.")
    print("  wider_lo/wider_hi are what the old 'wider of the two' rule gave,")
    print("  and hi_cut_% is how much narrower the export bound now is. A large")
    print("  cut means that zone's physical net position carries transit its")
    print("  commercial schedule never did - which the model cannot trade.")
    print(f"  wrote {PROCESSED / 'net_position.csv'}")
    print("\n  A zone whose lo and hi are far apart swings between importing and")
    print("  exporting; one with both bounds on the same side is structurally")
    print("  long or short and the constraint will bite in one direction only.")


if __name__ == "__main__":
    main()
