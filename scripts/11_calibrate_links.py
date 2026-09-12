"""Calibrate link capacities so simulated exchange matches observed exchange.

REJECTED. Kept because the negative result is worth more than the script.
=========================================================================
This works on its own terms and fails on the model's. With the hourly envelope
supplying the shape, scaling converges cleanly - eleven borders inside 13% of
observed exchange, most inside 5%, no oscillation. Flows match.

Prices then get worse, badly:

                       envelope only    + calibration     actual
    DE-FR mean              14.8             41.6          11.6
    DE-FR sd                29.3             85.5          34.4
    DE-PL mean               3.2             12.5           1.5
    NL sd ratio             2.44             5.59
    load shed             113 MWh        1,755 MWh

The reason is structural, and it is the same one that defeated the static
version from the other direction. Capacity must cover PEAK exchange; the
calibration targets MEAN exchange; a constant multiplier per border cannot
separate the two. Every scale factor that fixes an average removes headroom at
the peak, and in an import-dependent zone that headroom is what stops the
lights going out. Switzerland went first at 139 GWh of shed load, and once
Switzerland was excluded the Netherlands and Belgium went next.

The deeper point, which belongs in the write-up rather than in this file: on
DE-PL the auction traded 859 MW on average while hitting its limit in 2.8% of
hours, with prices that differed enough to give a spread standard deviation of
31 EUR/MWh. A market with large price differences that declines to trade to
its limit is telling you that allocated capacity in THOSE HOURS was small -
the flow-based domain closing DE-PL exactly when it mattered. Those domain
parameters are not published for CORE borders, so hourly allocated capacity
cannot be reconstructed from public data. The observed exchange series is the
only proxy available, and pushing it harder makes the model reproduce history
instead of responding to drivers.

So this correction is not applied. data/processed/link_scale.csv.rejected is
the output that was measured and dropped.


    python scripts/11_calibrate_links.py --days 30 --iters 8

Why this exists
---------------
The model is a transport model: each border carries its own independent limit,
so the optimiser can saturate all thirteen at once. The real market is
flow-based - borders compete for room on the same physical wires - so Europe
trades considerably less than a transport model wants to. Left uncorrected the
model moved 1.4x to 2.7x the observed power across every border, which
equalises prices and is why simulated spreads came out too narrow.

Shrinking the bilateral limits until simulated mean flow matches observed mean
flow is a crude stand-in for that shared constraint.

The target is FLOWS, not prices. Flows are an independent observable; fitting
capacities to prices would be circular, since prices are what the model is
being validated on.

Why this works now and did not before
-------------------------------------
The first version of this script scaled a STATIC per-border capacity, and it
could not work: a constant cap set so that mean flow matches observation must
bind in most hours, because the mean sits well below the peak. The model
congested borders 60-99% of the time where reality congested them 1-5%, and
the iteration oscillated (63% -> 47% -> 50% -> 65% -> 76%).

Capacity is now an hourly envelope derived from observed exchange, so its
SHAPE already follows when trade was possible. Scaling it changes how much
goes through a border without pinning that border at its limit - which is the
thing the static version could not separate.

The target is scheduled COMMERCIAL exchange, not physical flow: physical flow
includes loop flow the market never allocated, and on DE-PL that is the
difference between an 859 MW average and a 1,164 MW one.

One-sided by design
-------------------
Scale factors are capped at 1.0. Capacity is an upper bound: lowering it always
reduces flow, but raising it only increases flow on a border the model is
already saturating. Where the model trades LESS than reality, the cause is the
merit order, not the limit - the first version of this script chased those
borders to the ceiling and made the whole iteration oscillate (63% -> 47% ->
50% -> 65% -> 76% worst-case error over five rounds). Those borders are now
reported and left alone.

Writes data/processed/link_scale.csv, which the builder picks up automatically.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from spread.config import PROCESSED, load_config             # noqa: E402
from spread.network import build                             # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module                          # noqa: E402
load_inputs = import_module("10_build_network").load_inputs

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("pypsa", "linopy"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
log = logging.getLogger("calibrate")

DAMPING = 0.4      # the response of flow to capacity is nonlinear and the
                   # borders are coupled, so move only part of the way
# The floor is a feasibility guard, not a modelling statement. Calibrating a
# border down to match MEAN flow can leave it unable to serve a zone at PEAK,
# and an import-dependent zone then sheds load at VOLL - which is what happened
# to Switzerland at scale 0.455. A quiet border is allowed to shrink a long
# way; it is not allowed to disappear.
FLOOR, CEIL = 0.35, 1.0
BIND_TOL = 0.98    # "at capacity" for the binding diagnostic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--start", default="2025-01-13")
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    zones = cfg["zones"]

    # Borders touching a zone outside EU market coupling are NOT calibrated.
    #
    # This correction exists for one reason: a transport model gives every
    # border an independent limit and so over-allocates capacity that a
    # flow-based market shares. Switzerland is not in that market. Its borders
    # are NTC-based bilateral limits - a real, published constraint - so there
    # is nothing to correct, and scaling them to match a mean does active harm:
    # capacity must cover PEAK import, mean flow is far below peak, and a
    # constant multiplier cannot tell the two apart. Switzerland imports 3.4 GW
    # on average and needs nearly 6.8 GW at peak; calibrating to the average
    # shed 139 GWh of Swiss load at VOLL and dragged every neighbour with it.
    outside = set(cfg.get("outside_market_coupling", []))
    def _coupled(border: str) -> bool:
        a, _, b = border.partition(">")
        return a not in outside and b not in outside
    snapshots = pd.date_range(pd.Timestamp(args.start, tz="UTC"),
                              periods=args.days * 24, freq="h")
    inputs = load_inputs(snapshots)
    # The canonical exchange basis from scripts/02_build_hourly.py: commercial
    # everywhere it is the allocation mechanism, physical where it is not.
    # Reading raw commercial here instead cost 87 GWh of shed Swiss load.
    for name in ("exchange.parquet", "commercial_flows.parquet", "net_flows.parquet"):
        path = PROCESSED / name
        if path.exists():
            break
    log.info("calibrating against %s", path.name)
    observed = pd.read_parquet(path).reindex(snapshots)

    scale = {b: 1.0 for b in inputs["link_capacity"].index}
    last = None

    for it in range(1, args.iters + 1):
        inputs["link_scale"] = scale
        n = build(inputs, snapshots, zones)
        n.optimize(solver_name="highs",
                   solver_options={"threads": 1, "output_flag": False})

        rows = []
        for link in n.links.index:
            if link not in observed or not _coupled(link):
                continue
            obs_series = observed[link].dropna()
            if obs_series.empty:
                continue
            obs = float(obs_series.abs().mean())
            mod_series = n.links_t.p0[link]
            mod = float(mod_series.abs().mean())
            if obs <= 0 or mod <= 0:
                continue

            p_nom = float(n.links.at[link, "p_nom"])
            bind = float((mod_series.abs() >= BIND_TOL * p_nom).mean())
            ratio = obs / mod
            err = abs(ratio - 1)

            # sign agreement on the overlapping hours: does the model push
            # power the same way reality does?
            common = mod_series.set_axis(snapshots).reindex(obs_series.index)
            agree = float((np.sign(common) == np.sign(obs_series)).mean())

            rows.append({"link": link, "obs_mw": obs, "mod_mw": mod,
                         "err": err, "bind": bind, "agree": agree,
                         "scale": scale[link], "over": mod > obs})

            if mod > obs:                       # over-trading: shrink
                new = scale[link] * (1 + DAMPING * (ratio - 1))
                scale[link] = min(max(new, FLOOR), CEIL)
            # under-trading is a merit-order problem; capacity cannot fix it

        d = pd.DataFrame(rows).set_index("link")
        over = d[d["over"]]
        worst = float(over["err"].max()) if len(over) else 0.0
        mean_err = float(over["err"].mean()) if len(over) else 0.0
        log.info("iteration %d: over-trading on %d/%d borders, "
                 "worst %.1f%%, mean %.1f%%",
                 it, len(over), len(d), 100 * worst, 100 * mean_err)
        last = d
        if worst < 0.05:
            log.info("converged")
            break

    out = pd.DataFrame({"scale": pd.Series(scale)}).round(3)
    out.index.name = "border"
    out.to_csv(PROCESSED / "link_scale.csv")

    env = inputs.get("link_env_fwd")
    caps = inputs["link_capacity"]
    d = last.copy()
    d["scale"] = [round(scale[i], 3) for i in d.index]
    d["was_mw"] = [
        round(float(env[i].max()) if env is not None and i in env
              else float(caps.at[i, "p_nom_fwd_mw"]))
        for i in d.index
    ]
    d["now_mw"] = (d["was_mw"] * d["scale"]).round().astype(int)
    d["obs_mw"] = d["obs_mw"].round()
    d["mod_mw"] = d["mod_mw"].round()
    d["err_%"] = (100 * d["err"]).round(1)
    d["bind_%"] = (100 * d["bind"]).round(1)
    d["agree_%"] = (100 * d["agree"]).round(1)
    cols = ["obs_mw", "mod_mw", "err_%", "bind_%", "agree_%",
            "scale", "was_mw", "now_mw"]

    print("\n" + "=" * 78)
    print("OVER-TRADING BORDERS   (calibrated: capacity shrunk to match flow)")
    print("=" * 78)
    print(d[d["over"]][cols].sort_values("err_%", ascending=False).to_string())

    under = d[~d["over"]]
    if len(under):
        print("\n" + "=" * 78)
        print("UNDER-TRADING BORDERS   (left at 1.0 - capacity is not the constraint)")
        print("=" * 78)
        print(under[cols].sort_values("err_%", ascending=False).to_string())
        print("\n  These need a merit-order or availability explanation, not a")
        print("  smaller wire. bind_% near zero confirms the limit never binds.")

    skipped = [b for b in scale if not _coupled(b)]
    if skipped:
        print("\n  not calibrated (outside market coupling, NTC borders): "
              + ", ".join(sorted(skipped)))

    print(f"\nwrote {PROCESSED / 'link_scale.csv'}")
    print("Re-run scripts/10_build_network.py to see the effect on spreads.")


if __name__ == "__main__":
    main()
