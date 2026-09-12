"""Build the network and solve a test period. Reports time and memory.

    python scripts/10_build_network.py                 # one week
    python scripts/10_build_network.py --days 30
    python scripts/10_build_network.py --year 2025     # the full year

The first run's purpose is measurement, not results. Seconds per solve and
peak GB per solve decide the Monte Carlo design: how many workers fit in
16GB, how many draws are affordable, and whether the fleet needs aggregating
into blocks before package E.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd                                          # noqa: E402
import yaml                                                  # noqa: E402

from spread.config import PROCESSED, load_config, tee_output  # noqa: E402
from spread.blocks import aggregate_fleet, compare           # noqa: E402
from spread.process import unmodelled_generation             # noqa: E402
from spread.validate import (congestion_timing, net_position_binding,  # noqa: E402
                             price_separation, stopping_rule, supplementary)
from spread.network import (build, combine, core_net_position_limits,
                            derived_availability,
                            flow_based_constraints,  # noqa: E402
                            marginal_costs, net_position_limits,
                            hydro_reserve_credit, observed_availability_cap,
                            observed_net_position, reserve_constraint)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("pypsa").setLevel(logging.WARNING)
logging.getLogger("linopy").setLevel(logging.WARNING)
log = logging.getLogger("model")


def now_gb() -> float:
    """CURRENT resident memory, as opposed to the peak."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float("nan")


def peak_gb() -> float:
    """Peak resident memory. resource is POSIX-only, so Windows needs psutil."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1e9
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        except Exception:
            return float("nan")


def _maybe(path: Path):
    """Read a parquet if it exists, else None. Optional inputs stay optional."""
    return pd.read_parquet(path) if path.exists() else None


def _shocked_cf(shock: str | None, snapshots: pd.DatetimeIndex):
    """Capacity factors, optionally with renewable output scaled.

    THE FIRST VERSION OF THIS WAS REFUTED - see FINDINGS 13.8.
    It swapped a different calendar year's capacity factors onto this year's
    load and fuel prices. That moved the mean DE-FR spread by 16-22 EUR/MWh,
    more than any other driver, and the number was an artefact: it inflated the
    standard deviation from 28 to 196 while leaving the p90 and the MEDIAN
    untouched. Excluding hours where either price exceeded 500 EUR/MWh, the
    mean fell from 46.30 to 23.70 against a base of 23.45 - no effect at all.
    The run shed 176,540 MWh of load over 51 hours against 7 MWh in the base,
    and every zone hit VOLL. Swapping a year's renewable TIMING onto another
    year's demand, in a model with no storage and external borders frozen,
    simply manufactures scarcity.

    The replacement scales wind and solar output multiplicatively, which keeps
    each hour's shape and shifts only the resource level - the same kind of
    shock as a step in the gas price, and comparable to it.

    Hydro is deliberately excluded. The two years' run-of-river factors differ
    by far more than weather can explain (Switzerland 0.072 in 2024 against
    0.459 in 2025, a factor of 6.4), so that series carries a data defect and
    must not ride into a driver shock. It is recorded in FINDINGS section 10.
    """
    cf = pd.read_parquet(PROCESSED / "capacity_factors.parquet")
    if shock not in ("weather+", "weather-"):
        return cf

    wind_solar = [c for c in cf.columns
                  if any(k in str(c).lower() for k in ("wind", "solar"))]
    if not wind_solar:
        log.warning("SHOCK %s: no wind or solar columns found", shock)
        return cf

    # size the shock from the data: how much does the annual mean renewable
    # capacity factor actually move between the years on record?
    yearly = cf[wind_solar].groupby(cf.index.year).mean().mean(axis=1)
    yearly = yearly[yearly.index.isin([y for y in yearly.index
                                       if (cf.index.year == y).sum() > 8000])]
    if len(yearly) >= 2:
        rel = float(yearly.std() / yearly.mean())
    else:
        rel = 0.05
        log.warning("SHOCK %s: fewer than two full years, using 5%%", shock)
    sign = 1.0 if shock.endswith("+") else -1.0
    factor = 1.0 + sign * rel

    out = cf.copy()
    out[wind_solar] = (out[wind_solar] * factor).clip(0.0, 1.0)
    log.info("SHOCK %s: wind and solar capacity factors x%.4f "
             "(1 sd = %.1f%% of the mean, from annual means %s); "
             "hydro excluded - see the docstring",
             shock, factor, 100 * rel,
             {int(k): round(float(v), 4) for k, v in yearly.items()})
    return out


def load_inputs(snapshots: pd.DatetimeIndex, bands: int | None = None,
                bid_ladder: float | None = None,
                shock: str | None = None) -> dict:
    tech = yaml.safe_load(open(ROOT / "config" / "technology.yaml", encoding="utf-8"))
    if bid_ladder is not None:
        tech.setdefault("bid_ladder", {})["Nuclear"] = float(bid_ladder)
        log.info("bid_ladder override: Nuclear = %.1f EUR/MWh", bid_ladder)

    fleet = pd.read_csv(ROOT / "data" / "fleet" / "fleet.csv")
    fleet = fleet[fleet["capacity_mw"] > 0].copy()
    unit_fleet = fleet.copy()          # kept: outages are drawn at unit level
    fleet["gen_name"] = (fleet["zone"] + " " + fleet["tech"] + " "
                         + fleet.groupby(["zone", "tech"]).cumcount().astype(str))
    # A fuel with no traded series and no fixed cost used to fall through to
    # zero, which is how oil ended up burning free fuel for a month without a
    # single error. This project's characteristic failure is silence, so the
    # gap is now fatal rather than filled.
    TRADED = {"Natural Gas", "Hard Coal"}
    priced = set(tech["fixed_fuel_cost"]) | TRADED
    missing = sorted(set(fleet["fuel"]) - priced)
    if missing:
        raise SystemExit(
            f"fuels with no price: {missing}. Add them to fixed_fuel_cost in "
            "config/technology.yaml, or map them to a traded series in "
            "spread.network.marginal_costs. A fuel priced at zero is not a "
            "modelling choice, it is a bug."
        )
    fleet["fixed_fuel_cost"] = fleet["fuel"].map(tech["fixed_fuel_cost"]).fillna(0.0)

    # Collapse to dispatch blocks if asked. The unit fleet is kept alongside,
    # because driver 4 draws outages against real units and applies the result
    # to the block's available capacity - a 900 MW trip has to stay a 900 MW
    # step, not become a smooth derate across a band.
    if bands:
        fleet = aggregate_fleet(fleet, bands=bands)

    cf_report = pd.read_csv(PROCESSED / "cf_report.csv")
    res_p_nom = {
        r["column"]: (r["registered_mw"] if r["p_nom_source"] == "register"
                      else r["observed_peak_mw"])
        for _, r in cf_report.iterrows()
    }

    prices = pd.read_parquet(PROCESSED / "fuel_prices.parquet")

    # Driver shocks (FINDINGS 13.2).  The size is ONE STANDARD DEVIATION of
    # the daily series over the whole window the repo holds, computed here and
    # printed.  Deriving it from the data rather than choosing it stops the
    # magnitude being tuned until the comparison looks interesting - the same
    # discipline as the section 5 stopping rule.
    if shock in ("gas+", "gas-", "co2+", "co2-"):
        col = "gas_eur_mwh_th" if shock.startswith("gas") else "co2_eur_t"
        sd = float(prices[col].std())
        sign = 1.0 if shock.endswith("+") else -1.0
        before = float(prices.loc[prices.index.year.isin(snapshots.year.unique()),
                                  col].mean())
        prices[col] = (prices[col] + sign * sd).clip(lower=0.0)
        after = float(prices.loc[prices.index.year.isin(snapshots.year.unique()),
                                 col].mean())
        log.info("SHOCK %s: %s %+.2f (1 sd of the daily series), "
                 "mean %.2f -> %.2f", shock, col, sign * sd, before, after)

    scale_path = PROCESSED / "link_scale.csv"
    link_scale = (pd.read_csv(scale_path, index_col=0)["scale"].to_dict()
                  if scale_path.exists() else {})

    # Net-position bounds stand in for flow-based coupling: borders keep their
    # full rating, but a zone cannot export across all of them at once.
    np_path = PROCESSED / "net_position.csv"
    net_position = (pd.read_csv(np_path, index_col=0)
                    if np_path.exists() else None)
    # The constant bound stands in for a domain that is recomputed every hour.
    # If scripts/21_build_hourly_net_position.py has run, use the hourly one.
    nph_path = PROCESSED / "net_position_hourly.parquet"
    net_position_hourly = None
    if nph_path.exists():
        nph = pd.read_parquet(nph_path)
        if nph.index.tz is None:
            nph.index = nph.index.tz_localize("UTC")
        nph = nph.reindex(snapshots)
        # PyPSA drops the timezone, so index these the way the model will.
        nph.index = snapshots.tz_localize(None)
        net_position_hourly = nph
    generation = pd.read_parquet(PROCESSED / "generation.parquet")

    # Availability: a constant per technology, except where technology.yaml
    # says "derived", in which case it comes from observed generation.
    cap = pd.read_csv(PROCESSED / "installed_capacity.csv")
    # Match the register to the year being solved, not to the last year on
    # file - see the note in scripts/05_build_fleet.py. Availability is derived
    # against this, so the wrong year silently rescales a whole fleet.
    want = int(pd.Series(snapshots.year).mode().iloc[0])
    cap = cap[cap["year"] == min(cap["year"].unique(), key=lambda y: abs(y - want))]
    cap_lookup = {(r["zone"], r["tech"]): r["mw"] for _, r in cap.iterrows()}
    ENTSOE_TECH = {"Nuclear": "Nuclear"}

    zones = load_config()["zones"]
    availability = {}
    for tech_name, setting in tech.get("availability", {}).items():
        if setting == "derived":
            key = ENTSOE_TECH.get(tech_name, tech_name)
            availability[tech_name] = derived_availability(
                generation.reindex(snapshots), cap_lookup, key, zones)
        else:
            availability[tech_name] = float(setting)

    # Cap availability by observed output where technology.yaml asks for it.
    # This can only lower a constant, never raise it.
    obs_cfg = tech.get("availability_from_observed", {})
    if obs_cfg.get("enabled"):
        caps = observed_availability_cap(
            generation.reindex(snapshots), cap_lookup,
            obs_cfg.get("entsoe_series", {}), zones,
            {k: v for k, v in tech.get("availability", {}).items()
             if v != "derived"},
            window=int(obs_cfg.get("window_hours", 720)),
            floor=float(obs_cfg.get("floor", 0.45)),
        )
        availability.update(caps)

    if shock in ("frnuc+", "frnuc-"):
        # French nuclear availability is the driver section 13.4 flags as most
        # distorted by the fixed external boundary, so it is reported as a
        # separate sensitivity rather than folded into the decomposition.
        #
        # The magnitude used to be a hand-picked 10%, which made it the only
        # driver whose shock size was chosen rather than measured - and it came
        # out as 58% of the DE-FR decomposition, so that share rested on a
        # number I had invented. It is now derived the same way as gas, carbon
        # and weather: one standard deviation of the annual mean, across the
        # full years on record.
        #
        # WHAT THAT UNDERSTATES, and it matters: the window is 2024-2025, two
        # adjacent and unremarkable years. French nuclear produced about 279
        # TWh in 2022 during the stress-corrosion outages against roughly 360
        # TWh in 2024 - a swing far outside anything two good years can show.
        # So this is a lower bound on the driver's true variability, and the
        # same criticism applies to the weather shock. Section 13.9 records it.
        rel = 0.10
        try:
            gfr = None
            for c in generation.columns:
                key = "|".join(str(x) for x in c) if isinstance(c, tuple) else str(c)
                if "fr" in key.lower().split("|")[0].lower() and "nuclear" in key.lower():
                    gfr = generation[c]
                    break
            cap = {r["year"]: float(r["mw"]) for r in
                   __import__("csv").DictReader(
                       open(PROCESSED / "installed_capacity.csv"))
                   if r["zone"] == "FR" and r["tech"] == "Nuclear"}
            if gfr is not None and cap:
                cf = gfr.groupby(gfr.index.year).mean()
                full = [y for y in cf.index
                        if (gfr.index.year == y).sum() > 8000
                        and str(y) in cap and cap[str(y)] > 0]
                ann = {int(y): float(cf[y]) / cap[str(y)] for y in full}
                if len(ann) >= 2:
                    v = pd.Series(ann)
                    rel = float(v.std() / v.mean())
                    log.info("SHOCK %s: annual French nuclear availability %s, "
                             "1 sd = %.1f%% of the mean", shock,
                             {k: round(x, 3) for k, x in ann.items()},
                             100 * rel)
                else:
                    log.warning("SHOCK %s: fewer than two full years of French "
                                "nuclear, falling back to 10%%", shock)
        except Exception as exc:                              # noqa: BLE001
            log.warning("SHOCK %s: could not derive the magnitude (%s), "
                        "falling back to 10%%", shock, str(exc)[:60])
        step = rel * (1.0 if shock.endswith("+") else -1.0)
        cur = availability.get("Nuclear")
        if isinstance(cur, dict):
            if "FR" in cur:
                cur["FR"] = (cur["FR"] * (1.0 + step)).clip(0.0, 1.0) \
                    if hasattr(cur["FR"], "clip") else \
                    min(1.0, max(0.0, float(cur["FR"]) * (1.0 + step)))
                log.info("SHOCK %s: French nuclear availability x%.2f",
                         shock, 1.0 + step)
        elif cur is not None:
            availability["Nuclear"] = min(1.0, max(0.0, float(cur) * (1.0 + step)))
            log.info("SHOCK %s: nuclear availability x%.2f (all zones - the "
                     "config holds a scalar, not a per-zone series)",
                     shock, 1.0 + step)

    load_df = pd.read_parquet(PROCESSED / "load.parquet")
    unmodelled_pair = unmodelled_generation(
        load_df, generation, pd.read_parquet(PROCESSED / "net_flows.parquet"),
        zones, shapes=tech.get("unmodelled_shape", {}),
        bidding_share=tech.get("unmodelled_bidding_share", {}))

    # Operating reserve requirement: a share of load plus a share of variable
    # renewable output, which is how reserve is actually sized - wind and solar
    # forecast error is the thing being covered.
    res_cfg = tech.get("operating_reserve", {})
    requirement = None
    requirement_raw = None
    hydro_credit = None
    cm_zones = []
    if res_cfg.get("enabled"):
        gen_w = generation.reindex(snapshots)

        # PyPSA-Eur's epsilon_vres multiplies renewable POTENTIAL - capacity
        # times availability - not realised output, so the reserve held against
        # forecast error scales with what the fleet could have produced.
        cf_w = pd.read_parquet(PROCESSED / "capacity_factors.parquet").reindex(snapshots)
        vres = pd.DataFrame(0.0, index=snapshots, columns=zones)
        for col in cf_w.columns:
            zone, _, t = col.partition("|")
            if zone in zones and t in ("Solar", "Wind Onshore", "Wind Offshore"):
                vres[zone] = vres[zone] + (cf_w[col].fillna(0.0).clip(0, 1)
                                           * float(res_p_nom.get(col, 0.0)))

        ld = load_df.reindex(snapshots).ffill().bfill()
        cont = res_cfg.get("contingency_mw", {})
        if not isinstance(cont, dict):          # a scalar is applied to all
            cont = {z: float(cont) for z in zones}
        cont = pd.Series({z: float(cont.get(z, 0.0)) for z in zones})
        requirement = (float(res_cfg.get("epsilon_load", 0.0)) * ld[zones]
                       + float(res_cfg.get("epsilon_vres", 0.0)) * vres[zones]
                       + cont)
        requirement = requirement.fillna(0.0).clip(lower=0.0)
        # Keep the unscaled requirement so a sweep can re-apply its own pair of
        # multipliers without having to divide this one back out.
        requirement_raw = requirement.copy()
        cm_zones = [z for z in res_cfg.get("capacity_market_zones", []) if z in zones]
        mult = res_cfg.get("scarcity_multiplier", 1.0)
        if isinstance(mult, dict):
            cm = set(cm_zones)
            per_zone = pd.Series(
                {z: float(mult["capacity_market"] if z in cm else mult["default"])
                 for z in zones})
        else:
            per_zone = pd.Series({z: float(mult) for z in zones})
        requirement = requirement * per_zone
        hydro_credit = hydro_reserve_credit(gen_w, cap_lookup, zones).fillna(0.0)
        # PyPSA drops the timezone, so index these the way the model will index
        # them - otherwise a chunked solve cannot align them to its snapshots.
        naive = snapshots.tz_localize(None)
        requirement.index = naive
        requirement_raw.index = naive
        hydro_credit.index = naive

    return {
        "load": load_df,
        "reserve_requirement": requirement,
        "reserve_requirement_raw": requirement_raw,
        "capacity_market_zones": cm_zones,
        "reserve_hydro_credit": hydro_credit,
        "reserve_tiers": res_cfg.get("tiers", []),
        "thermal_names": pd.Index(fleet["gen_name"]),
        "capacity_factors": _shocked_cf(shock, snapshots),
        "generation": generation,
        "availability": availability,
        "must_run": tech.get("must_run", {}),
        "negative_bidding": tech.get("negative_bidding", {}),
        "negative_bidding_scale": tech.get("negative_bidding_scale", {}),
        "negative_bidding_share": tech.get("negative_bidding_share", {}),
        "unmodelled_bid": tech.get("unmodelled_bid", {}),
        "scarcity_tiers": tech.get("scarcity_tiers", []),
        "link_scale": link_scale,
        "unmodelled": unmodelled_pair[0],
        "unpublished": unmodelled_pair[1],
        "link_env_fwd": _maybe(PROCESSED / "link_env_fwd.parquet"),
        "link_env_rev": _maybe(PROCESSED / "link_env_rev.parquet"),
        "net_position": net_position,
        "net_position_hourly": net_position_hourly,
        "boundary": pd.read_parquet(PROCESSED / "boundary_position.parquet"),
        "link_capacity": pd.read_csv(PROCESSED / "link_capacity.csv", index_col=0),
        "fleet": fleet,
        "unit_fleet": unit_fleet,
        "res_p_nom": res_p_nom,
        "marginal_costs": marginal_costs(fleet, prices, snapshots,
                                         tech.get("bid_ladder", {})),
    }


def published_net_pos(year: int) -> pd.DataFrame:
    """JAO's maxNetPos as the "<zone>|hi" / "<zone>|lo" frame the model wants.

    These are CORE net positions. The caller must apply them to Core-internal
    links only and add the observed exchange with unrepresented Core zones -
    see core_net_position_limits().
    """
    path = ROOT / "data" / "raw" / "jao" / f"maxNetPos_{year}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} - fetch it with scripts/22")
    d = pd.read_csv(path, low_memory=False)
    d.columns = [c.strip().lstrip("\ufeff") for c in d.columns]
    d["t"] = pd.to_datetime(d["dateTimeUtc"], utc=True, errors="coerce")
    d = d.set_index("t")
    d = d[~d.index.duplicated(keep="first")]
    JAO = {"DE_LU": "DE", "FR": "FR", "PL": "PL", "NL": "NL",
           "BE": "BE", "AT": "AT", "CZ": "CZ"}
    out = pd.DataFrame(index=d.index)
    for zone, tag in JAO.items():
        if f"max{tag}" in d.columns and f"min{tag}" in d.columns:
            out[f"{zone}|hi"] = pd.to_numeric(d[f"max{tag}"], errors="coerce")
            out[f"{zone}|lo"] = pd.to_numeric(d[f"min{tag}"], errors="coerce")
    missing = [z for z in JAO if f"{z}|hi" not in out.columns]
    if missing:
        log.warning("  maxNetPos has no columns for %s - those zones will be "
                    "unconstrained", missing)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--start", default="2025-01-13")
    ap.add_argument("--shock", default=None,
                    choices=["gas+", "gas-", "co2+", "co2-",
                             "weather+", "weather-", "frnuc+", "frnuc-"],
                    help="apply one pre-committed driver shock (FINDINGS 13.2). "
                         "Magnitudes are one standard deviation of the daily "
                         "series over the whole available window, computed at "
                         "run time and printed, so they cannot be tuned to "
                         "produce an interesting answer.")
    ap.add_argument("--flow-based", nargs="?", const="auto", default=None,
                    help="use the real CORE constraints. Bare --flow-based "
                         "picks data/processed/fb_domain_<year>.parquet; give "
                         "a filename to override. Path under data/processed "
                         "scripts/29_build_domain.py, e.g. fb_domain_2024.parquet. "
                         "Replaces the net-position bounds with the real CORE "
                         "constraints. See FINDINGS section 13.")
    ap.add_argument("--max-net-pos", nargs="?", const="auto", default=None,
                    help="third constraint variant: JAO's PUBLISHED hourly "
                         "Core net-position limits, instead of the constant "
                         "bounds this repo estimates. Per zone like the "
                         "estimated bounds, hourly like the flow-based "
                         "domain, so it separates the two differences. Bare "
                         "flag reads data/raw/jao/maxNetPos_<year>.csv. "
                         "Mutually exclusive with --flow-based.")
    ap.add_argument("--slack-cost", type=float, default=5000.0,
                    help="EUR/MW penalty on violating a flow-based constraint. "
                         "Finite so the model always solves; report the slack.")
    ap.add_argument("--max-cnec", type=int, default=None,
                    help="keep at most this many constraints per hour, for "
                         "measuring how solve time scales")
    ap.add_argument("--bid-ladder", type=float, default=None,
                    help="override config bid_ladder for Nuclear, EUR/MWh. "
                         "A hand-edit of the yaml is easy to forget, and a "
                         "forgotten one produces two identical runs that look "
                         "like a comparison.")
    ap.add_argument("--tag", default=None,
                    help="name for this run's saved outputs under "
                         "data/processed/runs/. Defaults to the year.")
    ap.add_argument("--chunk-days", type=int, default=None,
                    help="solve in chunks of this many days. EXACT for this "
                         "model - it has no inter-temporal coupling - and cuts "
                         "peak memory roughly by the chunk factor")
    ap.add_argument("--ipm", action="store_true",
                    help="solve with HiGHS interior point instead of dual "
                         "simplex. Often lighter on time-coupled LPs; untested "
                         "here, and it changes which optimal basis you get, so "
                         "compare duals before trusting it")
    ap.add_argument("--bands", type=int, default=None,
                    help="aggregate the fleet into this many efficiency bands "
                         "per zone and technology. 3 gives roughly 150 blocks "
                         "from 865 units. Omit for the full unit-level fleet")
    args = ap.parse_args()

    # Per-tag log, not a single shared file. Two runs started in parallel
    # both tee into the same path otherwise, and their output interleaves
    # line by line - which produced a "net-position" log containing
    # flow-based constraint lines and cost figures inflated by CPU
    # contention. Nothing about the SOLUTION is affected (run outputs go to
    # data/processed/runs/<tag>/), but the diagnostics become unusable and
    # the corruption is not obvious at a glance.
    #
    # last_run.txt is still written, as the convenience alias for the most
    # recent run, so existing habits keep working.
    log_path = tee_output(f"run_{args.tag}" if args.tag else "last_run")

    cfg = load_config()
    zones = cfg["zones"]

    if args.year:
        snapshots = pd.date_range(f"{args.year}-01-01", f"{args.year + 1}-01-01",
                                  freq="h", tz="UTC", inclusive="left")
    else:
        start = pd.Timestamp(args.start, tz="UTC")
        snapshots = pd.date_range(start, periods=args.days * 24, freq="h", tz="UTC")

    log.info("snapshots: %d, %s to %s\n",
             len(snapshots), snapshots[0], snapshots[-1])

    t0 = time.time()
    inputs = load_inputs(snapshots, bands=args.bands,
                         bid_ladder=args.bid_ladder,
                         shock=args.shock)

    if args.flow_based and args.max_net_pos:
        raise SystemExit("--flow-based and --max-net-pos are different "
                         "constraint representations; choose one")

    fb_domain = fb_netpos = fb_outside = mnp_bounds = None
    if args.flow_based or args.max_net_pos:
        year = pd.Series(snapshots.year).mode().iloc[0]
        if args.max_net_pos:
            mnp_bounds = published_net_pos(year)
            log.info("max-net-pos: %s hourly rows from maxNetPos_%s.csv",
                     f"{len(mnp_bounds):,}", year)
        npf = ROOT / "data" / "raw" / "jao" / f"netPos_{year}.csv"
        if args.flow_based:
            name = (f"fb_domain_{year}.parquet" if args.flow_based == "auto"
                    else args.flow_based)
            fb_domain = pd.read_parquet(PROCESSED / name)
            fb_domain["t"] = pd.to_datetime(fb_domain["t"], utc=True)
            fb_domain = fb_domain[fb_domain["t"].isin(snapshots)]
            fb_netpos = pd.read_csv(npf, low_memory=False)
    if fb_netpos is not None:
        fb_netpos.columns = [c.strip().lstrip("\ufeff").lower()
                             for c in fb_netpos.columns]
        fb_netpos["t"] = pd.to_datetime(fb_netpos["datetimeutc"], utc=True,
                                        errors="coerce")
        fb_netpos = fb_netpos.set_index("t")
        fb_netpos = fb_netpos[~fb_netpos.index.duplicated(keep="first")]
    if args.flow_based or args.max_net_pos:
        # Core zones this model does not represent. JAO's net position
        # includes a zone's borders with them; this model has no such links,
        # so the observed exchange is added as a constant.
        OUTSIDE = ["HR", "HU", "RO", "SI", "SK"]
        sef = ROOT / "data" / "raw" / "jao" / f"scheduledExchanges_{year}.csv"
        fb_outside = None
        if sef.exists():
            se = pd.read_csv(sef, low_memory=False)
            se.columns = [c.strip().lstrip("\ufeff").lower()
                          for c in se.columns]
            se["t"] = pd.to_datetime(se["datetimeutc"], utc=True,
                                     errors="coerce")
            se = se.set_index("t")
            se = se[~se.index.duplicated(keep="first")]
            jao = {"DE_LU": "de", "FR": "fr", "PL": "pl", "NL": "nl",
                   "BE": "be", "AT": "at", "CZ": "cz"}
            fb_outside = pd.DataFrame(index=se.index)
            for z, tag in jao.items():
                tot = pd.Series(0.0, index=se.index)
                for w in OUTSIDE:
                    a, b = f"border_{tag}_{w.lower()}", f"border_{w.lower()}_{tag}"
                    if a in se.columns:
                        tot = tot + se[a].fillna(0.0)
                    if b in se.columns:
                        tot = tot - se[b].fillna(0.0)
                fb_outside[z] = tot
            nz = [c for c in fb_outside.columns if fb_outside[c].abs().mean() > 1]
            log.info("  outside-Core exchange added for %s (mean |MW|: %s)",
                     nz, {c: round(float(fb_outside[c].abs().mean()))
                          for c in nz})
        else:
            log.warning("  %s missing - outside-Core exchange NOT applied",
                        sef.name)

        if fb_domain is not None:
            log.info("flow-based: %s rows over %s hours, net positions from %s",
                     f"{len(fb_domain):,}", f"{fb_domain['t'].nunique():,}",
                     npf.name)
    log.info("inputs loaded in %.1fs  (%d thermal %s)",
             time.time() - t0, len(inputs["fleet"]),
             "blocks" if args.bands else "units")

    if args.bands:
        chk = compare(inputs["unit_fleet"], inputs["fleet"])
        drift = (chk["block_mw"] - chk["unit_mw"]).abs().max()
        print("\n" + "=" * 78)
        print("AGGREGATION CHECK   capacity must be preserved exactly")
        print("=" * 78)
        print(chk.dropna(how="all").to_string())
        print(f"\n  largest capacity drift: {drift:,.1f} MW  "
              f"({'ok' if drift < 1 else 'THIS IS A BUG'})")
        print("  eff_sd is what aggregation destroys: within-band efficiency")
        print("  spread. Where it was already near zero, nothing was lost.")

    mem_inputs = now_gb()

    t1 = time.time()
    n = build(inputs, snapshots, zones)
    t_build = time.time() - t1
    log.info("network built in %.1fs", t_build)
    log.info("  %d buses, %d generators, %d links, %d loads",
             len(n.buses), len(n.generators), len(n.links), len(n.loads))
    mem_build = now_gb()

    # The inputs dict holds several dense frames the network no longer needs -
    # generation, capacity factors, the raw unit fleet. In a Monte Carlo worker
    # that is dead weight held for the length of every solve. Dropped here and
    # measured, because the aggregation cut solve time ninefold and memory only
    # by half, which says the peak is no longer the LP.
    dropped = {k: inputs.pop(k) for k in
               ("generation", "capacity_factors", "unit_fleet", "boundary")
               if k in inputs}
    del dropped
    import gc
    gc.collect()
    mem_drop = now_gb()

    bounds = inputs.get("net_position")
    hourly_np = inputs.get("net_position_hourly")

    # The two arms of the FINDINGS section 13 study are mutually exclusive by
    # construction: net-position bounds are the crude stand-in FOR flow-based
    # coupling, so imposing both would double-count the same physics and make
    # the comparison meaningless.  --flow-based therefore REPLACES them.
    fb_slack_vars: list = []
    if fb_domain is not None:
        CORE = [z for z in zones if z != "CH"]
        fb_fn = flow_based_constraints(fb_domain, fb_netpos, zones,
                                       slack_cost=args.slack_cost,
                                       max_per_hour=args.max_cnec,
                                       sink=fb_slack_vars,
                                       core_zones=CORE,
                                       outside_core=fb_outside)
        # Switzerland is not in CORE, so no CNEC constrains it. Without its
        # net-position bound it would trade without limit, which is a
        # different model rather than a flow-based one.
        ch_fn = None
        if bounds is not None and "CH" in bounds.index:
            ch_fn = net_position_limits(bounds.loc[["CH"]], hourly_np)
        np_fn = combine(fb_fn, ch_fn)
        log.info("  CONSTRAINTS: flow-based on %s; net-position bound kept "
                 "for CH only", CORE)
    elif mnp_bounds is not None:
        # Per zone like the estimated bounds, hourly like the flow-based
        # domain. The difference from each isolates one of the two changes.
        CORE = [z for z in zones if z != "CH"]
        mnp_fn = core_net_position_limits(mnp_bounds, zones,
                                          core_zones=CORE,
                                          outside_core=fb_outside)
        ch_fn = None
        if bounds is not None and "CH" in bounds.index:
            ch_fn = net_position_limits(bounds.loc[["CH"]], hourly_np)
        np_fn = combine(mnp_fn, ch_fn)
        log.info("  CONSTRAINTS: JAO's published hourly net-position limits "
                 "on %s; estimated bound kept for CH only", CORE)
    else:
        np_fn = (net_position_limits(bounds, hourly_np)
                 if bounds is not None else None)
    if fb_domain is not None or mnp_bounds is not None:
        pass
    elif np_fn is None:
        log.info("  no net_position.csv - borders are independent "
                 "(run scripts/12_build_net_position.py)")
    elif hourly_np is None:
        log.info("  net position bounds are CONSTANT for the year "
                 "(run scripts/20 and 21 for the published hourly domain)")
    else:
        cov = [c.split("|")[0] for c in hourly_np.columns if c.endswith("|hi")]
        log.info("  net position bounds vary hourly for %d zones: %s",
                 len(cov), ", ".join(sorted(set(cov))))

    req = inputs.get("reserve_requirement")
    res_fn = (reserve_constraint(req, inputs["reserve_tiers"],
                                 inputs["thermal_names"],
                                 inputs.get("reserve_hydro_credit"))
              if req is not None and inputs.get("reserve_tiers") else None)
    extra = combine(np_fn, res_fn)

    t2 = time.time()
    opts = {"threads": 1, "output_flag": False}
    if args.ipm:
        opts["solver"] = "ipm"
        opts["run_crossover"] = "on"      # duals from a basic solution

    # Solve in chunks. This model has NO inter-temporal coupling - hydro is a
    # fixed profile, so there is no storage, and there are no ramp rates or
    # commitment constraints - so an 8,760-hour LP is 8,760 independent hourly
    # LPs stacked into one. Chunking is therefore EXACT, not an approximation,
    # and memory falls roughly with the chunk factor. The whole year peaked at
    # 6.8 GB, of which 6.4 GB was the solver.
    #
    # If inter-temporal constraints are ever added - real storage, ramping,
    # unit commitment - this becomes wrong and must be removed or given an
    # overlap. That is the one thing to remember about it.
    model_sns = n.snapshots
    if args.chunk_days:
        size = args.chunk_days * 24
        chunks = [model_sns[i:i + size] for i in range(0, len(model_sns), size)]
    else:
        chunks = [model_sns]

    # Reserve shortfall lives in the linopy model, and n.model only ever holds
    # the LAST one built - so under chunking the diagnostic would silently
    # report December instead of the year. Harvest it per chunk.
    reserve_names = [f"ReserveShortfall-{z}-{i}"
                     for z in zones
                     for i in range(len(inputs.get("reserve_tiers", [])))]
    reserve_parts = []

    fb_slack_parts: list = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            log.info("  chunk %d/%d: %s to %s", i, len(chunks),
                     chunk[0].date(), chunk[-1].date())
        n.optimize(snapshots=chunk, solver_name="highs", solver_options=opts,
                   extra_functionality=extra)

        if fb_domain is not None and fb_slack_vars:
            # Read through the reference handed back by the constraint
            # builder. Harvest per chunk, before the model is released -
            # section 6 records the bug where a chunked diagnostic silently
            # reported December only.
            try:
                fb_slack_parts.append(
                    np.asarray(fb_slack_vars[-1].solution.values).ravel())
            except Exception as exc:                              # noqa: BLE001
                log.warning("  slack unreadable: %s", str(exc)[:80])

        if reserve_names:
            present = [nm for nm in reserve_names if nm in n.model.variables]
            if present:
                reserve_parts.append(pd.DataFrame(
                    {nm: n.model.variables[nm].solution.to_pandas()
                     for nm in present}))

        # Release the model before building the next one. Thirteen monthly
        # chunks peaked at 4.08 GB where one month alone peaks at 0.91 - the
        # models were being held, not the results.
        if len(chunks) > 1:
            try:
                n._model = None
            except Exception:                                    # noqa: BLE001
                pass
            import gc
            gc.collect()

    reserve_sol = (pd.concat(reserve_parts).sort_index()
                   if reserve_parts else None)
    t_solve = time.time() - t2

    print("\n" + "=" * 70)
    print("TIMING AND SIZE   (this is what sizes the Monte Carlo)")
    print("=" * 70)
    print(f"  build            {t_build:8.1f} s")
    print(f"  solve            {t_solve:8.1f} s")
    print(f"  per 8760h solve  {t_solve * 8760 / len(snapshots):8.1f} s  (extrapolated)")
    print(f"  peak memory      {peak_gb():8.2f} GB")
    print(f"  resident: after inputs {mem_inputs:5.2f} | after build "
          f"{mem_build:5.2f} | after dropping input frames {mem_drop:5.2f} | "
          f"after solve {now_gb():5.2f} GB")
    print("  ^ where the peak lives decides how many Monte Carlo workers fit.")

    if "marginal_price" not in n.buses_t or n.buses_t.marginal_price.empty:
        print("\n  no prices - the solve did not reach optimality")
        return

    p = n.buses_t.marginal_price.copy()
    p.index = snapshots                      # reattach UTC, dropped for PyPSA
    actual = pd.read_parquet(PROCESSED / "prices.parquet").reindex(snapshots)

    # ------------------------------------------------------------------
    # Persist the run.  Until now the model reported summary statistics and
    # threw the series away, so any question the printed tables did not
    # already answer - which hours are wrong, what was marginal in them, how
    # the two distributions differ - needed a fresh 85-second solve and a new
    # print statement.  The 2x2 in FINDINGS section 13 compares five runs, so
    # the outputs have to survive the process that made them.
    #
    # --tag names the run, so cells do not overwrite each other:
    #     --tag ntc-8zone, --tag cnec-3zone, ...
    out = PROCESSED / "runs" / (args.tag or f"{args.year}")
    out.mkdir(parents=True, exist_ok=True)
    p.to_parquet(out / "prices.parquet")

    gen = getattr(n, "generators_t", None)
    if gen is not None and not gen.p.empty:
        g = gen.p.copy()
        g.index = snapshots
        # per zone and carrier, which is what a mix comparison needs
        carrier = n.generators.carrier.reindex(g.columns)
        zone_of = n.generators.bus.reindex(g.columns)
        by = g.T.groupby([zone_of, carrier]).sum().T
        by.columns = [f"{z}|{c}" for z, c in by.columns]
        by.to_parquet(out / "generation.parquet")

    if hasattr(n, "links_t") and not n.links_t.p0.empty:
        fl = n.links_t.p0.copy()
        fl.index = snapshots
        fl.to_parquet(out / "link_flows.parquet")

    # How hard did the solve lean on the violation slacks?  Section 13.5
    # makes this a stopping condition: if the model can only solve by breaking
    # the real constraints, they are not being imposed and nothing downstream
    # means anything.
    if fb_domain is not None:
        print("\n" + "=" * 70)
        print("FLOW-BASED CONSTRAINT VIOLATION")
        print("=" * 70)
        if not fb_slack_parts:
            print("  NO SLACK VARIABLE FOUND - the penalty may not have been")
            print("  applied, in which case violation was FREE and the")
            print("  constraints were not really imposed. Do not interpret")
            print("  this run until that is resolved.")
        else:
            sl = np.concatenate(fb_slack_parts)
            sl = sl[np.isfinite(sl)]
            print(f"  slack entries        {len(sl):12,}")
            print(f"  total slack          {sl.sum():12,.0f} MWh")
            print(f"  largest single       {sl.max():12,.0f} MW")
            print(f"  violated by >1 MW    {int((sl > 1.0).sum()):12,}"
                  f"  ({100*np.mean(sl > 1.0):.2f} %)")
            print(f"  penalty paid         {sl.sum()*args.slack_cost:12,.0f} EUR"
                  f"  at {args.slack_cost:.0f}/MW")
            print("  Near-zero means the real domain was imposed and the model")
            print("  lived inside it. Large means it could not, and section")
            print("  13.5 says stop rather than interpret the run.")

    print(f"\n  run saved to data/processed/runs/{out.name}/ "
          f"(prices, generation by zone and carrier, link flows)")

    # A spread is a difference of two levels. If each zone's own price is too
    # flat, no network fix can widen the spread - so the levels are checked
    # first, before anything is concluded about congestion.
    print("\n" + "=" * 78)
    print("ZONAL PRICE LEVELS, EUR/MWh   modelled vs observed")
    print("=" * 78)
    rows = []
    for zone in zones:
        if zone not in p or zone not in actual:
            continue
        m, a = p[zone].dropna(), actual[zone].dropna()
        rows.append({
            "zone": zone,
            "mod_mean": round(float(m.mean()), 1),
            "act_mean": round(float(a.mean()), 1),
            "mod_sd": round(float(m.std()), 1),
            "act_sd": round(float(a.std()), 1),
            "sd_ratio": round(float(m.std() / a.std()), 2) if a.std() else float("nan"),
            "mod_neg_h": int((m < 0).sum()),
            "act_neg_h": int((a < 0).sum()),
            "mod_hi_h": int((m > 150).sum()),
            "act_hi_h": int((a > 150).sum()),
        })
    print(pd.DataFrame(rows).set_index("zone").to_string())
    neg = pd.DataFrame(rows).set_index("zone")[["mod_neg_h", "act_neg_h"]]
    if neg["act_neg_h"].sum() or neg["mod_neg_h"].sum():
        print("\n  negative hours, modelled vs actual:  " + ",  ".join(
            f"{z} {r.mod_neg_h}/{r.act_neg_h}" for z, r in neg.iterrows()))
        print("  This is the check on negative_bidding - a COUNT, not a price.")
        shares = inputs.get("negative_bidding_share", {})
        if shares:
            print("  volume bidding below zero (negative_bidding_share):")
            for z in sorted(shares):
                sub = n.generators.index[n.generators.index.str.startswith(f"{z}|")]
                nosub = [g for g in sub if g.endswith(" unsubsidised")]
                if not nosub:
                    continue
                below = float(n.generators.p_nom[[g for g in sub
                                                  if not g.endswith(" unsubsidised")
                                                  and n.generators.marginal_cost[g] < 0]].sum())
                at_zero = float(n.generators.p_nom[nosub].sum())
                print(f"    {z}: {below:,.0f} MW below zero, "
                      f"{at_zero:,.0f} MW at zero "
                      f"({100 * below / max(below + at_zero, 1):.0f}% subsidised)")

    print("\n  sd_ratio well below 1 = the zone's own price formation is too flat,")
    print("  which caps the spread variance no matter what the borders do.")

    print("\n" + "=" * 70)
    print("SIMULATED SPREADS vs THE HISTORICAL TARGET")
    print("=" * 70)
    for a, b in [("DE_LU", "FR"), ("DE_LU", "PL")]:
        if a not in p or b not in p:
            continue
        sim = (p[a] - p[b]).dropna()
        act = (actual[a] - actual[b]).dropna()
        print(f"  {a}-{b}")
        print(f"    modelled   mean {sim.mean():7.2f}   sd {sim.std():6.2f}   "
              f"positive {100 * (sim > 0).mean():5.1f}%")
        print(f"    actual     mean {act.mean():7.2f}   sd {act.std():6.2f}   "
              f"positive {100 * (act > 0).mean():5.1f}%")

    # ---- the stopping rule, from spread.validate ----
    print("\n" + "=" * 78)
    print("STOPPING RULE   (design section 08, decided before package C began)")
    print("=" * 78)
    checks = stopping_rule(p, actual)
    width = max(len(c[0]) for c in checks)
    for name, value, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {value}")
    n_pass = sum(1 for _, _, ok in checks if ok)
    print(f"\n  {n_pass} of {len(checks)} criteria met.", end=" ")
    print("Package C is complete when all of them are."
          if n_pass < len(checks) else "Stopping rule MET - move to package D.")

    print("\n  supplementary, reported alongside the rule and never in place of it:")
    for name, value in supplementary(p, actual):
        print(f"    {name}  {value}")

    # ---- congestion: the mechanism the whole project turns on ----
    # A spread exists only when a link binds. If the model congests a border
    # far less often than reality does, its zones stay coupled and the spread
    # variance collapses - which is exactly the symptom on DE-PL.
    print("\n" + "=" * 70)
    print("CONGESTION: modelled vs observed, share of hours at capacity")
    print("=" * 70)
    # The observed congestion share used to be circular: p_nom was defined as
    # the 99th percentile of the same physical series, so ~1% exceedance was
    # guaranteed by construction and the column meant nothing. Commercial
    # exchange is an independent series - how often the AUCTION hit its limit -
    # so comparing the model against it is a real test.
    for _name in ("exchange.parquet", "commercial_flows.parquet", "net_flows.parquet"):
        _path = PROCESSED / _name
        if _path.exists():
            break
    obs_flows = pd.read_parquet(_path).reindex(snapshots)
    caps = inputs["link_capacity"]
    rows = []
    for link in n.links.index:
        p_nom = float(n.links.at[link, "p_nom"])
        modelled = n.links_t.p0[link].abs()
        mod_share = 100 * float((modelled >= 0.98 * p_nom).mean())

        obs_share = float("nan")
        if link in obs_flows:
            obs = obs_flows[link].abs().dropna()
            fwd = float(caps.at[link, "p_nom_fwd_mw"])
            rev = float(caps.at[link, "p_nom_rev_mw"])
            limit = max(fwd, rev)
            obs_share = 100 * float((obs >= 0.98 * limit).mean()) if limit else float("nan")

        rows.append({
            "link": link,
            "p_nom_mw": round(p_nom),
            "modelled_%": round(mod_share, 1),
            "observed_%": round(obs_share, 1),
            "mod_mean_mw": round(float(modelled.mean())),
            "obs_mean_mw": round(float(obs_flows[link].abs().mean())
                                 if link in obs_flows else float("nan")),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  modelled far below observed = the model keeps zones coupled that")
    print("  reality separates, and the spread variance goes missing there.")

    # ---- net position: the shared constraint that stands in for flow-based ----
    if inputs.get("net_position") is not None:
        print("\n" + "=" * 78)
        print("NET POSITION, MW   (positive = net exporter)")
        print("=" * 78)
        np_rows = []
        bounds = inputs["net_position"]
        obs_np = observed_net_position(obs_flows, zones)
        for zone in zones:
            if zone not in bounds.index:
                continue
            out_l = [k for k in n.links.index if n.links.at[k, "bus0"] == zone]
            in_l = [k for k in n.links.index if n.links.at[k, "bus1"] == zone]
            mod = (n.links_t.p0[out_l].sum(axis=1)
                   - n.links_t.p0[in_l].sum(axis=1))
            lo, hi = float(bounds.at[zone, "lo_mw"]), float(bounds.at[zone, "hi_mw"])
            span = max(hi - lo, 1.0)
            at_bound = ((mod >= hi - 0.01 * span) | (mod <= lo + 0.01 * span))
            obs = obs_np[zone].dropna() if zone in obs_np else pd.Series(dtype=float)
            np_rows.append({
                "zone": zone,
                "lo_mw": round(lo), "hi_mw": round(hi),
                "mod_mean": round(float(mod.mean())),
                "obs_mean": round(float(obs.mean())) if len(obs) else float("nan"),
                "mod_sd": round(float(mod.std())),
                "obs_sd": round(float(obs.std())) if len(obs) else float("nan"),
                "bound_%": round(100 * float(at_bound.mean()), 1),
            })
        print(pd.DataFrame(np_rows).set_index("zone").to_string())
        print("\n  bound_% is where the spread comes from: the constraint binds,")
        print("  the shadow price separates the zone from its neighbours. Reality")
        print("  separates zones a modest share of hours - neither 0% nor 90%.")

    # ---- operating reserve: how often headroom was scarce, and at what price ----
    if inputs.get("reserve_requirement") is not None and reserve_sol is not None:
        rows = []
        for zone in zones:
            names = [f"ReserveShortfall-{zone}-{i}"
                     for i in range(len(inputs["reserve_tiers"]))]
            names = [nm for nm in names
                     if reserve_sol is not None and nm in reserve_sol]
            if not names:
                continue
            sol = reserve_sol[names]
            short = sol.sum(axis=1)
            adder = pd.Series(0.0, index=short.index)
            for nm, tier in zip(names, inputs["reserve_tiers"]):
                adder = adder.where(sol[nm] <= 1.0, float(tier["price"]))
            rows.append({
                "zone": zone,
                "req_mean_mw": round(float(inputs["reserve_requirement"][zone].mean())),
                "short_hours_%": round(100 * float((short > 1).mean()), 1),
                "short_mean_mw": round(float(short.mean())),
                "max_adder": round(float(adder.max())),
            })
        if rows:
            print("\n" + "=" * 78)
            print("OPERATING RESERVE: how often headroom was scarce")
            print("=" * 78)
            print(pd.DataFrame(rows).set_index("zone").to_string())
            print("\n  short_hours_% is the share of hours the system could not hold its")
            print("  reserve target from spare thermal capacity. Those are the hours the")
            print("  scarcity adder lifts the price above the marginal unit's cost.")

    # ---- congestion TIMING, not just frequency ----
    flows = n.links_t.p0.copy()
    flows.index = snapshots
    timing = congestion_timing(
        flows, n.links["p_nom"].to_dict(), obs_flows)
    if not timing.empty:
        print("\n" + "=" * 78)
        print("CONGESTION TIMING: same hours, or just the same number of hours?")
        print("=" * 78)
        print(timing.to_string())
        print("\n  phi is the correlation of the two binary 'at capacity' series.")
        print("  hit_% is the share of genuinely congested hours the model also")
        print("  congests. Shares that agree with phi near zero means the model")
        print("  congests as OFTEN as reality and in DIFFERENT hours - which is")
        print("  what would leave spread distributions right and hourly")
        print("  correlation at 0.31.")
        for pair in ("DE_LU>FR", "DE_LU>PL"):
            if pair in timing.index:
                r = timing.loc[pair]
                print(f"    {pair}: phi {r['phi']:.3f}, catches {r['hit_%']:.0f}% "
                      f"of real congestion")
        print("\n  CAVEAT: this measures link p_nom, which net_position_limits()")
        print("  deliberately leaves slack. See the two tables below for the")
        print("  constraint that actually binds, and for the border-level test")
        print("  that does not depend on which constraint it is.")

    # ---- the constraint that actually binds ----
    if inputs.get("net_position") is not None:
        npb = net_position_binding(flows, n.links, inputs["net_position"],
                                   observed_net_position(obs_flows, zones))
        if not npb.empty:
            print("\n" + "=" * 100)
            print("NET POSITION BOUNDS: how often does the binding constraint bind?")
            print("=" * 100)
            print(npb.to_string())
            print("\n  bound_% is the share of hours the zone sat at its export or")
            print("  import limit. This - not link p_nom - is where the model's")
            print("  spreads come from, so a low number here means the model is")
            print("  producing spreads with almost nothing constraining it.")
            print("  obs_at_bound_% applies the SAME bounds to observed net")
            print("  positions: the bounds are annual quantiles of those flows, so")
            print("  the two columns should be of similar size if the model is")
            print("  trading the way Europe does.")

    # ---- the test that does not care which constraint binds ----
    borders = []
    for link in n.links.index:
        a, b = n.links.at[link, "bus0"], n.links.at[link, "bus1"]
        if a in zones and b in zones and (a, b) not in borders:
            borders.append((a, b))
    sep = price_separation(p, actual, borders)
    if not sep.empty:
        print("\n" + "=" * 100)
        print("PRICE SEPARATION: two coupled zones clear at ONE price unless "
              "something binds")
        print("=" * 100)
        print(sep.to_string())
        print("\n  Under implicit coupling |price difference| > 0.5 EUR IS")
        print("  congestion, observable on both sides with no NTC and no")
        print("  flow-based domain parameters. mod_% far below obs_% means the")
        print("  model has one price where reality has several, which caps every")
        print("  hourly spread statistic no matter how well the distributions")
        print("  match. Compare mod_% with obs_% FIRST, phi and hit_% second:")
        print("  separating too rarely and separating in the wrong hours are")
        print("  different defects with different fixes.")
        print("  if_random_% is what hit_% would be if the model's separated")
        print("  hours were drawn at random. hit_% at or near it means the")
        print("  model's congestion is statistically INDEPENDENT of reality's,")
        print("  which is a stronger statement than a low phi and is the one")
        print("  to quote.")

    shed = [g for g in n.generators.index if g.endswith(" shed")]
    shed_p = n.generators_t.p[shed]
    total_shed = shed_p.sum().sum()
    hours_shed = (shed_p.sum(axis=1) > 1).sum()
    print(f"\n  load shedding: {total_shed:,.0f} MWh over {hours_shed} hours "
          f"({100 * hours_shed / len(snapshots):.1f}% of the period)")
    if total_shed > 0:
        by_zone = shed_p.sum().sort_values(ascending=False)
        by_zone = by_zone[by_zone > 1]
        print("    by zone, MWh:  " + ",  ".join(
            f"{g.replace(' shed', '')} {v:,.0f}" for g, v in by_zone.items()))
    if hours_shed > 0.02 * len(snapshots):
        print("  ^ more than 2% - the model is short of capacity somewhere. "
              "Check before trusting prices.")


if __name__ == "__main__":
    try:
        main()
    finally:
        if hasattr(sys.stdout, "file"):
            print("\n[full output saved to logs/last_run.txt]")
            sys.stdout.flush()
