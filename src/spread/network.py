"""Assemble the PyPSA network from the processed inputs.

Design choices worth knowing before reading the code:

* Marginal costs step daily. They were monthly at first, on the theory that a
  coarser step saved memory - which was wrong: the frame is dense on snapshots
  either way, so the step costs nothing. It cost accuracy instead. TTF moved
  roughly 49 -> 58 -> 50 EUR/MWh inside February 2025, worth about +/-16
  EUR/MWh on a CCGT's SRMC, and a monthly mean erases all of it.

* Hydro follows history. Reservoir and pumped storage enter as fixed profiles
  rather than optimised storage, because an LP with perfect foresight over a
  year of inflows dispatches water far better than any real operator can, and
  that error would land straight on the German and French borders. The cost is
  that hydro cannot respond to the drivers.

* Every zone gets a load-shedding generator at VOLL. Not a trick to avoid
  errors - it is what a real system does when short, and it stops Monte Carlo
  draws failing selectively in exactly the scarcity hours the study is about.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

log = logging.getLogger(__name__)

VOLL = 3000.0                       # EUR/MWh, value of lost load
HYDRO_PROFILE_TECHS = ["Hydro Water Reservoir", "Hydro Pumped Storage"]


def marginal_costs(fleet: pd.DataFrame, prices: pd.DataFrame,
                   snapshots: pd.DatetimeIndex,
                   bid_ladder: dict | None = None) -> pd.DataFrame:
    """SRMC per unit per snapshot: fuel/eff + co2 intensity/eff x co2 + VOM.

    Steps daily and is reindexed onto the snapshots. Within-day shape is not
    modelled: a plant's fuel is bought forward, so intraday gas moves do not
    reach the day-ahead offer, but day-to-day moves certainly do.
    """
    daily = prices.resample("D").mean().ffill()

    fuel_col = {"Natural Gas": "gas_eur_mwh_th", "Hard Coal": "coal_eur_mwh_th"}
    out = {}
    for unit in fleet.itertuples():
        fuel_price = (daily[fuel_col[unit.fuel]]
                      if unit.fuel in fuel_col
                      else pd.Series(unit.fixed_fuel_cost, index=daily.index))
        srmc = (fuel_price / unit.efficiency
                + daily["co2_eur_t"] * unit.co2_t_per_mwh_th / unit.efficiency
                + unit.vom_eur_per_mwh)
        out[unit.gen_name] = srmc

    frame = pd.DataFrame(out)

    # bid_ladder: turn a flat technology block into a rising offer curve.
    # Units of the same technology in the same zone are ordered largest
    # first and given an adder running from 0 to `spread` across their
    # cumulative capacity, so the marginal unit's price depends on how deep
    # into the fleet demand reaches.  See config/technology.yaml.
    for tech, spread in (bid_ladder or {}).items():
        if not spread:
            continue
        sel = fleet[fleet["tech"] == tech]
        for zone, grp in sel.groupby("zone"):
            grp = grp.sort_values(["capacity_mw", "gen_name"],
                                  ascending=[False, True])
            total = grp["capacity_mw"].sum()
            if total <= 0:
                continue
            cum = 0.0
            for unit in grp.itertuples():
                mid = cum + unit.capacity_mw / 2.0
                if unit.gen_name in frame.columns:
                    frame[unit.gen_name] = frame[unit.gen_name] + \
                        float(spread) * (mid / total)
                cum += unit.capacity_mw

    frame.index = frame.index.tz_convert(snapshots.tz)
    return (frame.reindex(frame.index.union(snapshots))
            .ffill().bfill().reindex(snapshots).astype("float32"))


def derived_availability(generation: pd.DataFrame, capacity: dict,
                         tech: str, zones: list[str],
                         window: int = 168) -> dict:
    """Availability for a baseload technology, from a rolling max of output.

    Nuclear runs flat out when it can, so the highest output over a recent
    window is close to what was actually available. That recovers the real
    outage pattern - seasonal maintenance, the French summer dip - without
    needing to reconstruct it from the outage event tables.
    """
    out = {}
    for zone in zones:
        col = f"{zone}|{tech}"
        if col not in generation:
            continue
        p_nom = float(capacity.get((zone, tech), 0.0))
        if p_nom <= 0:
            continue
        series = generation[col].fillna(0.0).clip(lower=0)
        rolling = (series.rolling(window, min_periods=1, center=True).max()
                   / p_nom).clip(0.05, 1.0)
        out[zone] = rolling.ffill().bfill()
    return out


def build(inputs: dict, snapshots: pd.DatetimeIndex,
          zones: list[str]) -> pypsa.Network:
    """Assemble the network. `inputs` is the dict from load_inputs()."""
    n = pypsa.Network()

    # PyPSA refuses timezone-aware snapshots, so the tz is dropped here and
    # only here. Everything upstream stays UTC, every series below is passed
    # as .values so alignment is positional rather than by index, and the
    # caller reattaches the UTC index to the results. Converting to local
    # time anywhere in this pipeline is how you lose an hour in March.
    n.set_snapshots(snapshots.tz_localize(None))

    for zone in zones:
        n.add("Bus", zone)

    # ---- demand ----
    load = inputs["load"].reindex(snapshots)
    for zone in zones:
        if zone in load:
            n.add("Load", f"{zone} load", bus=zone,
                  p_set=load[zone].ffill().bfill().values)

    # ---- net exchange with the unmodelled world ----
    # A positive boundary position is an export, which the zone must cover:
    # it enters as extra load. Negative becomes a must-take import.
    pos = inputs["boundary"].reindex(snapshots).fillna(0.0)
    for zone in zones:
        if zone not in pos:
            continue
        series = pos[zone]
        if series.abs().max() < 1:
            continue
        n.add("Load", f"{zone} boundary export", bus=zone,
              p_set=series.clip(lower=0).values)
        imp = (-series).clip(lower=0)
        if imp.max() > 0:
            n.add("Generator", f"{zone} boundary import", bus=zone,
                  p_nom=float(imp.max()), marginal_cost=0.0,
                  p_min_pu=(imp / imp.max()).values,
                  p_max_pu=(imp / imp.max()).values)

    # ---- generation this model does not represent ----
    # See unmodelled_generation() in spread.process. Two things at once: real
    # output published under ENTSO-E types this model does not build (4.4 GW of
    # Dutch "Other" against 1 MW of registered capacity), and output nobody
    # published at all (small Swiss hydro). Together 45% of Dutch load.
    for label, frame, default_bid in (
            # The flat floor: industrial, CHP and waste-gas output that runs
            # through the night. It does not pay to run, so it bids zero.
            ("unmodelled flat", inputs.get("unmodelled"), 0.0),
            # Everything above that floor: behind-the-meter solar, which loses
            # its subsidy when curtailed and so bids below zero.
            ("unmodelled solar", inputs.get("unpublished"), None)):
        if frame is None:
            continue
        for zone in zones:
            if zone not in frame:
                continue
            series = frame[zone].reindex(snapshots).fillna(0.0)
            peak = float(series.max())
            if peak <= 0:
                continue
            # CURTAILABLE, not forced. This was p_min_pu = p_max_pu, which was
            # harmless while the profile was a smooth 1.5 GW daily average and
            # fatal once it carried a real solar shape: forced Dutch generation
            # exceeded load plus export capacity at summer midday, and since the
            # model can absorb a shortage (load shedding) but not a surplus, the
            # nodal balance had no solution and the whole year went infeasible.
            #
            # The negative bid is what makes it run, not a lower bound - the
            # same way every other renewable here self-dispatches. That leaves
            # it able to curtail in the hours where it would otherwise break the
            # problem, which is also what the real plant does.
            bid = (default_bid if default_bid is not None
                   else float(inputs.get("unmodelled_bid", {}).get(zone, 0.0)))
            n.add("Generator", f"{zone} {label}", bus=zone,
                  carrier=label, p_nom=peak, marginal_cost=bid,
                  p_max_pu=(series / peak).values)

    # ---- variable renewables and run-of-river ----
    cf = inputs["capacity_factors"].reindex(snapshots)
    for col in cf.columns:
        zone, _, tech = col.partition("|")
        if zone not in zones:
            continue
        p_nom = float(inputs["res_p_nom"].get(col, 0.0))
        if p_nom <= 0:
            continue
        # Subsidised plant bids below zero rather than curtail: the premium is
        # lost on every MWh not produced. Without this the price floor is zero
        # and the model cannot reproduce a negative hour - 457 of them in
        # Germany in 2024. See negative_bidding in config/technology.yaml.
        bid = (float(inputs.get("negative_bidding", {}).get(tech, 0.0))
               * float(inputs.get("negative_bidding_scale", {}).get(zone, 1.0)))

        # Not all of a zone's renewable fleet is paid to run. A plant outside
        # a premium - past its support term, merchant or PPA, or suspended by
        # a negative-price rule - curtails at zero rather than paying to
        # generate. Bid DEPTH cannot represent that: the negative-hour count
        # is a step function of whether ANYTHING bids below zero, so it is
        # identical at -4 and -8. Only the VOLUME split moves the count, which
        # is why this is a share of capacity and not a shift in the bid.
        # See negative_bidding_share in config/technology.yaml for the source.
        profile = cf[col].fillna(0.0).clip(0, 1).values
        share = float(inputs.get("negative_bidding_share", {})
                      .get(zone, {}).get(tech, 1.0))
        share = min(max(share, 0.0), 1.0)
        if share >= 1.0 or bid == 0.0:
            n.add("Generator", f"{col}", bus=zone, carrier=tech, p_nom=p_nom,
                  marginal_cost=bid, p_max_pu=profile)
        else:
            # Same bus, same carrier, same profile - every downstream
            # aggregation groups on those, not on the generator name.
            n.add("Generator", f"{col}", bus=zone, carrier=tech,
                  p_nom=p_nom * share, marginal_cost=bid, p_max_pu=profile)
            n.add("Generator", f"{col} unsubsidised", bus=zone, carrier=tech,
                  p_nom=p_nom * (1.0 - share), marginal_cost=0.0,
                  p_max_pu=profile)

    # ---- hydro that follows history ----
    gen = inputs["generation"].reindex(snapshots)
    for zone in zones:
        for tech in HYDRO_PROFILE_TECHS:
            col = f"{zone}|{tech}"
            if col not in gen:
                continue
            series = gen[col].fillna(0.0).clip(lower=0)
            peak = float(series.max())
            if peak <= 0:
                continue
            # Hydro is deliberately NOT given a negative bid: reservoir and
            # pumped storage have an opportunity cost, not a subsidy, and
            # run-of-river spills rather than pays to run.
            n.add("Generator", col, bus=zone, carrier=tech, p_nom=peak,
                  marginal_cost=0.0, p_max_pu=(series / peak).values)

    # ---- thermal fleet ----
    fleet = inputs["fleet"]
    costs = inputs["marginal_costs"].reindex(snapshots)
    avail = inputs["availability"]              # {tech: float or {zone: Series}}
    must_run = inputs["must_run"]               # {tech: share of available}

    for unit in fleet.itertuples():
        if unit.zone not in zones:
            continue
        setting = avail.get(unit.tech, 1.0)
        if isinstance(setting, dict):           # derived, per zone and hour
            profile = setting.get(unit.zone)
            p_max_pu = (profile.values if profile is not None else 1.0)
        else:
            p_max_pu = float(setting)

        # Must-run is a share of what is AVAILABLE, not of nameplate - a unit
        # on outage has no minimum. Getting that wrong makes the problem
        # infeasible the moment availability drops below the must-run level.
        share = must_run.get(unit.tech, 0.0)
        p_min_pu = (p_max_pu * share if share else 0.0)

        n.add("Generator", unit.gen_name, bus=unit.zone, carrier=unit.tech,
              p_nom=float(unit.capacity_mw),
              p_max_pu=p_max_pu, p_min_pu=p_min_pu,
              marginal_cost=costs[unit.gen_name].values)

    # ---- interconnectors ----
    caps = inputs["link_capacity"]
    for border, row in caps.iterrows():
        a, b = border.split(">")
        if a not in zones or b not in zones:
            continue                       # boundary borders are fixed flows
        # Calibration factor from scripts/11_calibrate_links.py. A transport
        # model has one independent limit per border, so it can saturate all
        # of them at once; the real flow-based domain makes borders compete
        # for the same physical wires, so Europe trades far less. Scaling the
        # bilateral limits until simulated exchange matches observed exchange
        # is a crude stand-in for that shared constraint - calibrated against
        # flows, which are observed, not against prices, which are the target.
        scale = float(inputs.get("link_scale", {}).get(border, 1.0))

        # Preferred: an hourly envelope from observed exchange, because
        # allocated capacity is not a constant - see link_envelope() in
        # spread.process. Falls back to the static percentile if absent.
        env_f = inputs.get("link_env_fwd")
        env_r = inputs.get("link_env_rev")
        if (env_f is not None and env_r is not None
                and border in env_f and border in env_r):
            f = env_f[border].reindex(snapshots).ffill().bfill().to_numpy() * scale
            r = env_r[border].reindex(snapshots).ffill().bfill().to_numpy() * scale
            p_nom = float(max(f.max(), r.max()))
            if p_nom <= 0:
                continue
            n.add("Link", border, bus0=a, bus1=b, p_nom=p_nom,
                  p_max_pu=f / p_nom, p_min_pu=-r / p_nom)
            continue

        fwd = float(row["p_nom_fwd_mw"]) * scale
        rev = float(row["p_nom_rev_mw"]) * scale
        p_nom = max(fwd, rev)
        if p_nom <= 0:
            continue
        n.add("Link", border, bus0=a, bus1=b, p_nom=p_nom,
              p_max_pu=fwd / p_nom, p_min_pu=-rev / p_nom)

    # ---- demand-side response and reserves ----
    for zone in zones:
        if zone not in load:
            continue
        peak = float(load[zone].max())
        for i, tier in enumerate(inputs.get("scarcity_tiers", [])):
            p_nom = peak * float(tier["share_of_peak"])
            if p_nom <= 0:
                continue
            n.add("Generator", f"{zone} DSR {i}", bus=zone, carrier="DSR",
                  p_nom=p_nom, marginal_cost=float(tier["price"]))

    # ---- load shedding ----
    for zone in zones:
        n.add("Generator", f"{zone} shed", bus=zone, carrier="load shedding",
              p_nom=1e5, marginal_cost=VOLL)

    return n


def flow_based_constraints(domain: pd.DataFrame,
                           net_positions: pd.DataFrame,
                           zones: list[str],
                           slack_cost: float = 5000.0,
                           max_per_hour: int | None = None,
                           sink: list | None = None,
                           core_zones: list[str] | None = None,
                           outside_core: pd.DataFrame | None = None):
    """The real CORE constraints: SUM PTDF x NetPosition <= RAM.

    Returns an extra_functionality, the flow-based counterpart to
    net_position_limits() above.  That function bounds each zone's net
    position one zone at a time, which is the crudest possible stand-in for
    this; here every constraint is a named network element under a named
    contingency, with the transmission system operators' own PTDFs and
    remaining margin.

    THE CONSTRAINT
    --------------
    For each critical network element and contingency, in each hour:

        SUM over all twelve Core zones of PTDF(zone) x NP(zone)  <=  RAM

    The sum runs over zones this model does not solve for, so their terms move
    to the right-hand side using what those zones actually did:

        SUM over MODELLED zones  <=  RAM - SUM over FIXED zones

    `domain` supplies both sides: ptdf_<zone> columns for the modelled zones,
    fixed_<zone> for the rest, and `ram`.  `net_positions` supplies the
    observed hourly figures for the fixed zones.

    VALIDATED BEFORE USE
    --------------------
    The formulation is not assumed.  scripts/32 and 37 tested five RAM
    definitions, both sign conventions and four net-position sources against
    the constraints JAO reports as actually binding.  The winner - published
    RAM, published sign, Switzerland excluded, net positions from JAO's own
    netPos - puts binding constraints at a median margin of ZERO against a
    slack median of 804 MW, and reproduces them to within 50 MW for 85% of the
    economically significant ones (shadow price above 196 EUR/MW).

    WHY THE ARRAYS LOOK THE WAY THEY DO
    -----------------------------------
    The constraint set changes every hour: about 105 of them, but not the same
    105.  So this cannot be one broadcast constraint.  Instead the rows are
    packed into a dense (snapshot x slot) grid - slot 0..N-1 within each hour -
    padded where an hour has fewer.  Padding gets a PTDF of zero and an
    enormous RAM, so a padded row is satisfied by anything and costs the
    solver nothing.

    FEASIBILITY
    -----------
    A real published domain imposed on a model whose fundamentals are off can
    have NO solution: the net positions the dispatch wants may sit outside the
    polytope.  That would show up as a crash, not a bad number.  So every
    constraint carries a non-negative slack priced at `slack_cost` per MW -
    high enough that violation is a last resort, finite so the model always
    solves.  Report the slack: if it is large the constraints are not really
    being imposed, and section 13.5 makes that a stopping condition.
    """
    def _add(n, snapshots):
        p = n.model["Link-p"]
        dim = next(d for d in p.dims if d != "snapshot")

        # Net position of each modelled zone, as a linear expression.
        #
        # It must be the CORE net position, which is what JAO's PTDFs and RAM
        # are defined against - not the zone's total exchange. Two corrections
        # follow from that, and the first run without them put Switzerland at
        # EUR 323 against an actual EUR 40.
        #
        #   Only CORE-INTERNAL links count. Switzerland is not a Core zone, so
        #   DE->CH is invisible to the flow-based domain. Counting it inflates
        #   the German net position, consumes CNEC headroom JAO never
        #   allocated, and starves Switzerland.
        #
        #   Borders to Core zones this model does not represent are ADDED as a
        #   constant. JAO's hub_AT includes Austria-Hungary and
        #   Austria-Slovenia; this model has no such links, so without the
        #   observed exchange the Austrian net position is only partial and the
        #   constraint is applied to the wrong quantity.
        core = set(core_zones or zones)
        np_expr = {}
        for zone in zones:
            if zone not in core:
                continue
            out_l = [k for k in n.links.index
                     if n.links.at[k, "bus0"] == zone
                     and n.links.at[k, "bus1"] in core]
            in_l = [k for k in n.links.index
                    if n.links.at[k, "bus1"] == zone
                    and n.links.at[k, "bus0"] in core]
            e = None
            if out_l:
                e = p.sel({dim: out_l}).sum(dim)
            if in_l:
                imp = p.sel({dim: in_l}).sum(dim)
                e = -imp if e is None else e - imp
            if e is not None:
                np_expr[zone] = e
        if not np_expr:
            print("    flow-based: NO LINKS FOUND - constraint not applied")
            return

        # PyPSA builds the network on timezone-NAIVE snapshots (load_inputs
        # strips the tz so a chunked solve can align them), while the domain
        # carries tz-aware UTC. Comparing the two matches nothing, and the
        # first version of this returned silently on the empty result - three
        # runs looked like they had applied the constraints and had not.
        # Match on a common representation, and never fail quietly again.
        snaps = pd.DatetimeIndex(snapshots)
        model_naive = snaps.tz is None
        dom_key = (domain["t"].dt.tz_localize(None) if model_naive
                   else domain["t"])
        d = domain[dom_key.isin(snaps)].copy()
        if d.empty:
            print(f"    flow-based: NO OVERLAP between the domain "
                  f"({domain['t'].min()} .. {domain['t'].max()}) and the "
                  f"model snapshots ({snaps.min()} .. {snaps.max()}) - "
                  f"CONSTRAINT NOT APPLIED")
            return
        d = d.sort_values(["t", "cid"])
        d["_k"] = (d["t"].dt.tz_localize(None) if model_naive else d["t"])

        # slot index within each hour, then a dense (snapshot x slot) grid
        d = d.assign(slot=d.groupby("_k").cumcount())
        n_slot = int(d["slot"].max()) + 1
        if max_per_hour:
            d = d[d["slot"] < max_per_hour]
            n_slot = min(n_slot, max_per_hour)
        tpos = {t: i for i, t in enumerate(snaps)}
        ti = d["_k"].map(tpos).to_numpy()
        si = d["slot"].to_numpy()

        BIG = 1e7
        rhs = np.full((len(snaps), n_slot), BIG, dtype="float64")
        coef = {z: np.zeros((len(snaps), n_slot), dtype="float64")
                for z in np_expr}

        # right-hand side: RAM less what the zones we do not model contribute
        r = d["ram"].to_numpy(dtype="float64").copy()
        for col in [c for c in d.columns if c.startswith("fixed_")]:
            zone = col.split("_", 1)[1]
            src = f"hub_{zone.lower()}"
            if src not in net_positions.columns:
                continue
            idx_np = d["t"]
            if getattr(net_positions.index, "tz", None) is None:
                idx_np = idx_np.dt.tz_localize(None)
            npv = net_positions[src].reindex(idx_np).to_numpy(dtype="float64")
            r = r - d[col].to_numpy(dtype="float64") * np.nan_to_num(npv)
        rhs[ti, si] = r

        for z in np_expr:
            col = f"ptdf_{z}"
            if col in d.columns:
                coef[z][ti, si] = d[col].to_numpy(dtype="float64")

        # move the observed non-modelled-Core exchange onto the right-hand
        # side: PTDF x (link flows + constant) <= RAM  becomes
        # PTDF x link flows <= RAM - PTDF x constant
        if outside_core is not None:
            oc = outside_core
            idx_oc = d["t"]
            if getattr(oc.index, "tz", None) is None:
                idx_oc = idx_oc.dt.tz_localize(None)
            adj = np.zeros(len(d), dtype="float64")
            for z in np_expr:
                if z not in oc.columns or f"ptdf_{z}" not in d.columns:
                    continue
                cval = oc[z].reindex(idx_oc).to_numpy(dtype="float64")
                adj = adj + d[f"ptdf_{z}"].to_numpy(dtype="float64") \
                    * np.nan_to_num(cval)
            rhs[ti, si] = rhs[ti, si] - adj

        cdim = "cnec"
        slots = np.arange(n_slot)
        lhs = None
        for z, e in np_expr.items():
            da = xr.DataArray(coef[z], dims=["snapshot", cdim],
                              coords={"snapshot": snaps, cdim: slots})
            term = e * da
            lhs = term if lhs is None else lhs + term

        slack = n.model.add_variables(
            lower=0.0,
            coords=[pd.Index(snaps, name="snapshot"),
                    pd.Index(slots, name=cdim)],
            name="FlowBased-slack")

        n.model.add_constraints(
            lhs - slack <= xr.DataArray(rhs, dims=["snapshot", cdim],
                                        coords={"snapshot": snapshots,
                                                cdim: slots}),
            name="FlowBased-cnec")

        # Attach the penalty.  Depending on the linopy version `objective`
        # is either an expression or an Objective wrapper, so try the
        # in-place add first and fall back to reassignment.
        pen = (slack_cost * slack).sum()
        attached = "?"
        try:
            n.model.objective += pen
            attached = "+="
        except Exception:                                     # noqa: BLE001
            try:
                n.model.objective = n.model.objective + pen
                attached = "="
            except Exception as exc:                          # noqa: BLE001
                attached = f"FAILED {str(exc)[:50]}"

        # Hand the caller a direct reference.  Looking the variable up by
        # name after the solve returned a KeyError and iterating the
        # container returned nothing, so do not rely on either.
        if sink is not None:
            sink.append(slack)

        print(f"    flow-based: {len(d):,} constraints over "
              f"{d['t'].nunique()} hours, up to {n_slot} per hour, "
              f"slack at {slack_cost:.0f} EUR/MW, penalty attached [{attached}]")

    return _add


def net_position_limits(bounds: pd.DataFrame, hourly: pd.DataFrame | None = None):
    """Cap each zone's net internal exchange. Returns an extra_functionality.

    Why not per-border capacity
    ---------------------------
    A transport model gives every border its own independent limit, so the
    optimiser can saturate all thirteen at once and trades far more power than
    Europe really does. The obvious repair - shrink each border until its mean
    flow matches observation - fails for a structural reason: a static limit
    that equals MEAN flow must bind in most hours, because the mean sits well
    below the peak. The model then congests borders 60-99% of the time where
    reality congests them 1-5%, which is the same error with the sign flipped.

    Flow-based coupling does not limit borders one at a time. It limits the
    COMBINATION of exchanges, because every trade loads the same physical
    wires. The minimal representation of that in a transport model is a bound
    on each zone's net position: borders keep their full thermal capacity and
    stay uncongested most hours, but the zone cannot export across all of them
    simultaneously. When the bound binds, the shadow price separates that zone
    from its neighbours - which is where the spread comes from.

    Bounds come from observed net positions (scripts/12_build_net_position.py),
    so this is calibrated against flows, an independent observable, not against
    prices, which are what the model is being validated on.

    Hourly bounds
    -------------
    Pass `hourly` - a frame of "<zone>|hi" and "<zone>|lo" columns indexed by
    snapshot, from scripts/21_build_hourly_net_position.py - and the bound
    varies by hour instead of standing still all year. That matters because the
    thing it stands in for, the CORE flow-based domain, is recomputed every
    hour, and a constant bound can only ever congest in hours the model chooses
    for itself. price_separation() showed the consequence: the model's
    congestion is statistically independent of reality's on all thirteen
    borders.

    A zone or an hour missing from `hourly` falls back to the constant. Never
    to nothing - Switzerland is not in CORE and must keep a bound.
    """
    def _add(n, snapshots):
        p = n.model["Link-p"]
        # linopy names the component dimension after the index, which PyPSA
        # calls "name", not "Link". Find it rather than hard-coding it.
        dim = next(d for d in p.dims if d != "snapshot")

        def rhs(zone, side, fallback):
            """Constant, or an hourly series aligned to THIS chunk."""
            col = f"{zone}|{side}"
            if hourly is None or col not in hourly:
                return float(fallback)
            # The solve is chunked, so slice to the snapshots handed in and
            # fill any gap with the constant rather than dropping the bound.
            series = hourly[col].reindex(snapshots).astype(float)
            if series.isna().all():
                return float(fallback)
            series = series.fillna(float(fallback))
            return xr.DataArray(series.values, dims=["snapshot"],
                                coords={"snapshot": snapshots})

        varying = 0
        for zone, row in bounds.iterrows():
            out = [k for k in n.links.index if n.links.at[k, "bus0"] == zone]
            inn = [k for k in n.links.index if n.links.at[k, "bus1"] == zone]
            expr = None
            if out:
                expr = p.sel({dim: out}).sum(dim)
            if inn:
                imports = p.sel({dim: inn}).sum(dim)
                expr = -imports if expr is None else expr - imports
            if expr is None:
                continue
            hi = rhs(zone, "hi", row["hi_mw"])
            lo = rhs(zone, "lo", row["lo_mw"])
            varying += int(not isinstance(hi, float))
            n.model.add_constraints(expr <= hi, name=f"NetPosition-max-{zone}")
            n.model.add_constraints(expr >= lo, name=f"NetPosition-min-{zone}")
        log.info("net position limits applied to %d zones (%d hourly)",
                 len(bounds), varying)
    return _add


def core_net_position_limits(hourly: pd.DataFrame,
                             zones: list[str],
                             core_zones: list[str] | None = None,
                             outside_core: pd.DataFrame | None = None):
    """JAO's OWN published hourly net-position limits, one box per Core zone.

    The third constraint representation, sitting between the other two.

        net_position_limits()       a box this repo estimated, constant all year
        core_net_position_limits()  the box JAO publishes, redrawn every hour
        flow_based_constraints()    the polytope itself

    Why this exists
    ---------------
    The other two differ in TWO ways at once - per zone against per element,
    and constant against hourly - so neither difference can be attributed.
    This variant is per zone AND hourly, which separates them: against
    net_position_limits() it measures the value of hourly variation alone,
    against flow_based_constraints() the value of the polytope alone.

    It also answers the obvious objection to the comparison, that the
    per-zone representation is one this repo invented. These bounds are
    JAO's.

    SCOPE - the error this must not repeat
    --------------------------------------
    JAO's maxNetPos is a zone's CORE net position, across its Core borders
    only. It is NOT the zone's total exchange. Applying it to the total would
    charge German exports to Switzerland against an allowance that never
    covered them - the same mistake that priced Switzerland at EUR 323 against
    an actual EUR 40 when the flow-based constraints were first built. So the
    expression here counts CORE-INTERNAL links only, and borders with Core
    zones the model does not represent enter as an observed constant, exactly
    as in flow_based_constraints().

    Switzerland is not in Core and gets no bound from here; the caller must
    keep its estimated one.
    """
    def _add(n, snapshots):
        p = n.model["Link-p"]
        dim = next(d for d in p.dims if d != "snapshot")
        core = set(core_zones or zones)
        snaps = pd.DatetimeIndex(snapshots)
        model_naive = snaps.tz is None

        def align(frame):
            """Match the model's tz convention, then slice to this chunk."""
            if frame is None:
                return None
            f = frame.copy()
            tz = getattr(f.index, "tz", None)
            if model_naive and tz is not None:
                f.index = f.index.tz_localize(None)
            elif (not model_naive) and tz is None:
                f.index = f.index.tz_localize("UTC")
            return f.reindex(snaps)

        h = align(hourly)
        oc = align(outside_core)
        if h is None or h.dropna(how="all").empty:
            print("    max-net-pos: NO OVERLAP between the published bounds "
                  f"and the model snapshots ({snaps.min()} .. {snaps.max()}) "
                  "- CONSTRAINT NOT APPLIED")
            return

        applied, filled = [], 0
        for zone in zones:
            if zone not in core:
                continue
            hi_col, lo_col = f"{zone}|hi", f"{zone}|lo"
            if hi_col not in h.columns or lo_col not in h.columns:
                continue

            out_l = [k for k in n.links.index
                     if n.links.at[k, "bus0"] == zone
                     and n.links.at[k, "bus1"] in core]
            in_l = [k for k in n.links.index
                    if n.links.at[k, "bus1"] == zone
                    and n.links.at[k, "bus0"] in core]
            expr = None
            if out_l:
                expr = p.sel({dim: out_l}).sum(dim)
            if in_l:
                imp = p.sel({dim: in_l}).sum(dim)
                expr = -imp if expr is None else expr - imp
            if expr is None:
                continue

            hi = pd.to_numeric(h[hi_col], errors="coerce").astype(float)
            lo = pd.to_numeric(h[lo_col], errors="coerce").astype(float)

            # Borders with Core zones the model does not represent are part of
            # the published net position but not of the expression, so move
            # them to the bound instead of leaving the constraint on the wrong
            # quantity.
            if oc is not None and zone in oc.columns:
                c = oc[zone].reindex(snaps).fillna(0.0).astype(float)
                hi = hi - c.values
                lo = lo - c.values

            # A missing hour must widen the bound, never tighten it: a NaN
            # filled with a small number would invent congestion.
            gaps = int(hi.isna().sum() + lo.isna().sum())
            if gaps:
                filled += gaps
                hi = hi.fillna(hi.max() if hi.notna().any() else 1e6)
                lo = lo.fillna(lo.min() if lo.notna().any() else -1e6)

            hi_da = xr.DataArray(hi.values, dims=["snapshot"],
                                 coords={"snapshot": snapshots})
            lo_da = xr.DataArray(lo.values, dims=["snapshot"],
                                 coords={"snapshot": snapshots})
            n.model.add_constraints(expr <= hi_da,
                                    name=f"CoreNetPosition-max-{zone}")
            n.model.add_constraints(expr >= lo_da,
                                    name=f"CoreNetPosition-min-{zone}")
            applied.append(zone)

        if not applied:
            print("    max-net-pos: NO ZONES MATCHED - CONSTRAINT NOT APPLIED")
            return
        log.info("published Core net-position limits applied hourly to "
                 "%d zones: %s%s", len(applied), ", ".join(applied),
                 f" ({filled} missing values widened)" if filled else "")
    return _add


def observed_net_position(net_flows: pd.DataFrame, zones: list[str]) -> pd.DataFrame:
    """Net internal exchange per zone per hour, from observed border flows.

    Sign convention matches the model: positive is a net export.
    """
    out = {}
    for zone in zones:
        total = None
        for border in net_flows.columns:
            a, _, b = border.partition(">")
            if a not in zones or b not in zones:
                continue                      # boundary borders are not links
            if zone == a:
                term = net_flows[border]
            elif zone == b:
                term = -net_flows[border]
            else:
                continue
            total = term if total is None else total + term
        if total is not None:
            out[zone] = total
    return pd.DataFrame(out)


def observed_availability_cap(generation: pd.DataFrame, capacity: dict,
                              series_map: dict, zones: list[str],
                              constants: dict, window: int = 720,
                              floor: float = 0.45) -> dict:
    """Cap each technology's availability by what it was observed to produce.

    The diagnostic that motivated this: in the tightest 5% of hours the model
    was carrying 26 GW of spare thermal capacity in Germany against 70 GW of
    load, almost all of it gas. Real German gas never reached those levels in
    that month at any price. Capacity that did not respond to a 300 EUR/MWh
    price signal is not available capacity, whatever the register says - the
    same principle already applied to the Swiss register.

    The rolling maximum of observed output over a long window is a floor on
    what was available, not a measurement of it: a plant out of merit all
    month looks unavailable. So the window is deliberately long enough to
    contain at least one tight event, a floor stops a quiet month collapsing
    the fleet, and the result can only ever REDUCE the constant in
    technology.yaml, never raise it. It is a cap, not an estimate.
    """
    out = {}
    for tech, entsoe in series_map.items():
        const = float(constants.get(tech, 1.0))
        per_zone = {}
        for zone in zones:
            col = f"{zone}|{entsoe}"
            if col not in generation:
                continue
            p_nom = float(capacity.get((zone, entsoe), 0.0))
            if p_nom <= 0:
                continue
            observed = generation[col].fillna(0.0).clip(lower=0)
            ratio = (observed.rolling(window, min_periods=1, center=True).max()
                     / p_nom)
            capped = np.minimum(const, np.maximum(ratio, floor))
            per_zone[zone] = pd.Series(capped, index=generation.index).ffill().bfill()
        if per_zone:
            out[tech] = per_zone
    return out


def hydro_reserve_credit(generation: pd.DataFrame, capacity: dict,
                         zones: list[str],
                         techs: list[str] | None = None) -> pd.DataFrame:
    """Headroom on reservoir and pumped storage, MW per zone per hour.

    Hydro enters the model as a fixed historical profile, so PyPSA sees no
    headroom on it at all - available equals dispatched by construction. That
    is fine for energy and wrong for reserve: a reservoir running at 30% of
    its turbine capacity is exactly what a TSO holds reserve on, and in
    Switzerland, Austria and France it is most of the reserve there is.

    Without this the reserve constraint was short in 100% of Swiss hours and
    85% of French ones - not because those systems were tight, but because
    the only plant allowed to hold reserve was thermal.

    The credit is registered turbine capacity (floored at observed peak, since
    registers disagree with reality often enough) minus observed output. It is
    a constant per hour, not a decision, so it moves to the right-hand side.
    """
    techs = techs or HYDRO_PROFILE_TECHS
    out = {}
    for zone in zones:
        total = None
        for tech in techs:
            col = f"{zone}|{tech}"
            if col not in generation:
                continue
            series = generation[col].fillna(0.0).clip(lower=0)
            p_nom = max(float(capacity.get((zone, tech), 0.0)), float(series.max()))
            if p_nom <= 0:
                continue
            head = (p_nom - series).clip(lower=0)
            total = head if total is None else total + head
        out[zone] = (total if total is not None
                     else pd.Series(0.0, index=generation.index))
    return pd.DataFrame(out)


def reserve_constraint(requirement: pd.DataFrame, tiers: list[dict],
                       thermal: pd.Index,
                       fixed_headroom: pd.DataFrame | None = None):
    """An operating reserve demand curve. Returns an extra_functionality.

    Why the model has no top to its price distribution
    --------------------------------------------------
    Cost-based LP dispatch prices energy at the short-run cost of the marginal
    unit. That reproduced the bottom three quartiles of the 2025 price
    distribution well and none of the top: modelled prices stopped at about
    127 EUR/MWh, one CCGT's SRMC, in a month whose 99th percentile was 314.

    That gap is not a calibration error, it is what cost-based dispatch does.
    Real markets pay scarcity rent for headroom long before they physically
    run out of plant - through reserve procurement, through the risk premium
    in an offer, and through the fact that the last MWh in a tight hour is
    bid, not costed. A production cost model reproduces it with an operating
    reserve demand curve: the system must hold headroom, and the price of
    headroom rises in steps as it becomes scarce. The dual of that constraint
    adds to the energy price, which is the mechanism being represented.

    This is a calibrated behavioural mechanism, not a fundamental one, and it
    should be labelled that way in anything written up. It carries three
    numbers. They are set against 2025 and must be validated on 2024
    unchanged - a curve refitted per year would be fitting noise.
    """
    def _add(n, snapshots):
        m = n.model
        p = m["Generator-p"]
        dim = next(d for d in p.dims if d != "snapshot")
        snap = p.indexes["snapshot"]
        avail = (n.get_switchable_as_dense("Generator", "p_max_pu")
                 .mul(n.generators.p_nom, axis=1))

        for zone in requirement.columns:
            gens = [g for g in thermal if g in n.generators.index
                    and n.generators.at[g, "bus"] == zone]
            if not gens:
                continue
            # Slice to the snapshots THIS call was given, not the whole year:
            # the network is solved in chunks, and extra_functionality is
            # handed only the chunk. Taking .to_numpy() on the full series
            # would silently misalign every constraint after the first chunk.
            req = requirement[zone].reindex(snap).to_numpy(dtype=float)

            # headroom = available - dispatched. The available part is a
            # constant per snapshot, so it moves to the right-hand side and
            # only the dispatch variables stay on the left.
            head_const = avail.loc[snap, gens].sum(axis=1).to_numpy(dtype=float)
            if fixed_headroom is not None and zone in fixed_headroom:
                head_const = head_const + (fixed_headroom[zone].reindex(snap)
                                           .to_numpy(dtype=float))
            expr = -p.sel({dim: gens}).sum(dim)

            for i, tier in enumerate(tiers):
                share = float(tier["share_of_requirement"])
                slack = m.add_variables(
                    lower=0.0, upper=req * share, coords=[snap],
                    name=f"ReserveShortfall-{zone}-{i}",
                )
                expr = expr + slack
                m.objective = m.objective + (float(tier["price"]) * slack).sum()

            rhs = xr.DataArray(req - head_const, coords=[snap], dims=["snapshot"])
            m.add_constraints(expr >= rhs, name=f"OperatingReserve-{zone}")

        log.info("operating reserve applied to %d zones, %d tiers",
                 len(requirement.columns), len(tiers))
    return _add


def combine(*functions):
    """Chain several extra_functionality callbacks into one."""
    live = [f for f in functions if f is not None]
    if not live:
        return None

    def _add(n, snapshots):
        for f in live:
            f(n, snapshots)
    return _add
