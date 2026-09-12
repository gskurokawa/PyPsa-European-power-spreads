"""Turn cached raw chunks into aligned hourly frames.

Three things this has to get right, all discovered the hard way in package A:

1. Resolution varies by series AND changes mid-series. Austrian generation is
   quarter-hourly where Swiss is hourly; FR-DE flows switch from hourly to
   quarter-hourly partway through April 2025, so one monthly file contains
   both. The fix is to resample unconditionally - .resample("h").mean() is
   correct whatever the input granularity, and detection logic would only
   add bugs.

2. Chunks overlap. entsoe-py pads each request by a day either side, so
   adjacent monthly files share timestamps. Concatenating without
   deduplicating silently double-counts the boundary hours.

3. Gaps must be visible. Reindexing onto a complete hourly range turns a
   missing month into explicit NaN rather than absent rows, so the coverage
   report can find it instead of the model quietly running on less data
   than you think.

Everything stays in UTC. Market-time conversion is a presentation concern.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .entsoe_pull import load_raw

log = logging.getLogger(__name__)

TZ = "UTC"


def hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    """Complete hourly UTC index over [start, end)."""
    return pd.date_range(
        pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ),
        freq="h", inclusive="left", name="timestamp",
    )


def to_hourly(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deduplicate, resample to hourly means, reindex onto the full range."""
    if df.empty:
        return pd.DataFrame(index=index)

    df = df[~df.index.duplicated(keep="first")].sort_index()
    numeric = df.select_dtypes("number")
    if numeric.empty:
        return pd.DataFrame(index=index)

    return numeric.resample("h").mean().reindex(index)


def build_zone_series(dataset: str, column: str, raw_dir: Path,
                      index: pd.DatetimeIndex) -> pd.DataFrame:
    """One column per zone, for single-series datasets (load, prices)."""
    out = {}
    for zone, df in load_raw(dataset, raw_dir).items():
        hourly = to_hourly(df, index)
        if hourly.empty:
            continue
        # single-series datasets come back with one numeric column
        out[zone] = hourly.iloc[:, 0]
    frame = pd.DataFrame(out, index=index)
    frame.columns.name = "zone"
    return frame.rename(columns=str)


# ENTSO-E production-type names are not stable over time. Anything here is a
# rename the publisher made mid-series; without this they arrive as two
# half-empty columns that look like missing data.
TECH_ALIASES = {
    "Hydro Run-of-river and poundage": "Hydro Run-of-river and pondage",
}


def _tech_name(col: str) -> str | None:
    """Technology for a generation column, or None if the column is consumption.

    Two shapes appear in the same series. Where a zone reports both generation
    and consumption for a technology, entsoe-py returns a MultiIndex that the
    puller flattened to 'Tech | Actual Aggregated'. Where it reports generation
    only - as Poland did until mid-2024 - the column is a bare 'Tech'.

    Filtering on 'Actual Aggregated' alone silently drops every bare column,
    which is how five months of Polish generation disappeared without a single
    error being raised.
    """
    if "|" in col:
        tech, _, qualifier = col.partition("|")
        tech, qualifier = tech.strip(), qualifier.strip()
        if qualifier and qualifier != "Actual Aggregated":
            return None                      # consumption - a v2 concern
    else:
        tech = col.strip()
    return TECH_ALIASES.get(tech, tech)


def build_generation(raw_dir: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Columns are 'ZONE|Technology'. Generation only; consumption dropped."""
    frames = []
    for zone, df in load_raw("generation", raw_dir).items():
        hourly = to_hourly(df, index)
        if hourly.empty:
            continue

        keep = {c: t for c in hourly.columns if (t := _tech_name(c)) is not None}
        if not keep:
            continue
        hourly = hourly[list(keep)]
        hourly.columns = list(keep.values())

        # A technology can appear under more than one name across the period
        # (bare vs suffixed, or a spelling the publisher corrected). Those
        # variants are disjoint in time, so max() merges them without
        # double-counting.
        if hourly.columns.duplicated().any():
            merged = sorted({c for c in hourly.columns[hourly.columns.duplicated()]})
            log.info("    %s: merged duplicate technology names: %s",
                     zone, ", ".join(merged))
            hourly = hourly.T.groupby(level=0).max().T

        hourly.columns = [f"{zone}|{c}" for c in hourly.columns]
        frames.append(hourly)

    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=index)


def build_net_flows(raw_dir: Path, index: pd.DatetimeIndex,
                    borders: list[str]) -> pd.DataFrame:
    """Net flow per border: flow(A->B) minus flow(B->A).

    ENTSO-E publishes physical flow as two separate non-negative series, so
    neither direction alone is the net. Positive means A exports to B.
    """
    raw = load_raw("flows", raw_dir)
    out = {}
    for border in borders:
        a, b = border.split(">")
        fwd = raw.get(f"{a}>{b}")
        rev = raw.get(f"{b}>{a}")
        if fwd is None and rev is None:
            continue
        f = to_hourly(fwd, index).iloc[:, 0] if fwd is not None else pd.Series(0.0, index=index)
        r = to_hourly(rev, index).iloc[:, 0] if rev is not None else pd.Series(0.0, index=index)
        out[border] = f.fillna(0) - r.fillna(0)
    frame = pd.DataFrame(out, index=index)
    frame.columns.name = "border"
    return frame


def build_commercial_flows(raw_dir: Path, index: pd.DatetimeIndex,
                           borders: list[str]) -> pd.DataFrame:
    """Net day-ahead SCHEDULED COMMERCIAL exchange per border.

    Same shape as build_net_flows, different series, and the difference is a
    modelling question rather than a data-cleaning one.

    Physical flow is what crossed the wire. Scheduled commercial exchange is
    what was traded across the border in the day-ahead auction. They diverge
    wherever loop flow is large: power sold from one German zone to another
    physically routes through Poland and Czechia, occupying transmission that
    the market coupling never had to allocate. Sizing a market model from
    physical flow therefore hands the optimiser capacity the auction did not
    own, and over-couples exactly those borders.
    """
    raw = load_raw("scheduled", raw_dir)
    out = {}
    for border in borders:
        a, b = border.split(">")
        fwd, rev = raw.get(f"{a}>{b}"), raw.get(f"{b}>{a}")
        if fwd is None and rev is None:
            continue
        f = to_hourly(fwd, index).iloc[:, 0] if fwd is not None else pd.Series(0.0, index=index)
        r = to_hourly(rev, index).iloc[:, 0] if rev is not None else pd.Series(0.0, index=index)
        out[border] = f.fillna(0) - r.fillna(0)
    frame = pd.DataFrame(out, index=index)
    frame.columns.name = "border"
    return frame


def blend_exchange(commercial: pd.DataFrame, physical: pd.DataFrame,
                   outside: list[str]) -> pd.DataFrame:
    """Commercial exchange per border, except where it is not the allocation.

    Borders touching a zone outside EU market coupling take physical flow.
    See outside_market_coupling in config/zones.yaml for why.
    """
    out = {}
    for border in commercial.columns.union(physical.columns):
        a, _, b = border.partition(">")
        use_physical = a in outside or b in outside
        src = physical if use_physical else commercial
        if border in src:
            out[border] = src[border]
    frame = pd.DataFrame(out)
    frame.columns.name = "border"
    return frame


def link_envelope(exchange: pd.DataFrame, window: int = 168,
                  floor_share: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly capacity per border per direction, from observed exchange.

    Allocated cross-border capacity is not a constant. Under flow-based
    coupling the room on a border depends on how the rest of the grid is
    loaded, so what the auction can allocate on DE-PL swings from a few
    hundred MW to over two gigawatts within a week.

    A single percentile cannot represent that, and the symptom is specific:
    with p_nom fixed at the 99th percentile of commercial exchange, the model
    congested DE-PL in 38.6% of hours where the auction hit its limit in 2.8%,
    while simultaneously trading 1,643 MW on average against an observed 859.
    Both at once - too often at the cap, and too much through it - is what a
    constant looks like when the real quantity varies.

    So capacity is derived the same way plant availability now is: a rolling
    maximum of what was observed, per direction. The window is a week, long
    enough that the model is not pinned to the exact hour history traded; the
    floor keeps a quiet week from closing a border entirely.

    Returns two frames of MW, forward and reverse, indexed by snapshot.
    """
    fwd, rev = {}, {}
    for border in exchange.columns:
        s = exchange[border].fillna(0.0)
        f = s.clip(lower=0)
        r = (-s).clip(lower=0)
        for series, target in ((f, fwd), (r, rev)):
            roll = series.rolling(window, min_periods=1, center=True).max()
            floor = float(series.quantile(0.99)) * floor_share
            target[border] = roll.clip(lower=floor)
    return (pd.DataFrame(fwd, index=exchange.index),
            pd.DataFrame(rev, index=exchange.index))


def link_capacity(net_flows: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    """Link p_nom from observed flow, since CORE borders publish no NTC.

    Taken per direction, because interconnectors are not always symmetric.
    """
    rows = []
    for border in net_flows.columns:
        s = net_flows[border].dropna()
        if s.empty:
            continue
        rows.append({
            "border": border,
            "p_nom_fwd_mw": round(float(s.clip(lower=0).quantile(quantile)), 1),
            "p_nom_rev_mw": round(float((-s).clip(lower=0).quantile(quantile)), 1),
            "max_fwd_mw": round(float(s.max()), 1),
            "max_rev_mw": round(float(-s.min()), 1),
            "hours": int(s.size),
        })
    return pd.DataFrame(rows).set_index("border")


def coverage(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """How much of each column is actually populated."""
    n = len(frame)
    rows = [{
        "dataset": name,
        "column": col,
        "hours": n,
        "missing": int(frame[col].isna().sum()),
        "pct_complete": round(100 * (1 - frame[col].isna().mean()), 2),
    } for col in frame.columns]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# capacity factors and boundary positions
# --------------------------------------------------------------------------

# Technologies that enter the model as an availability profile rather than a
# dispatchable unit. p_max_pu for these comes from observed generation over
# installed capacity.
RES_TECHS = [
    "Wind Onshore",
    "Wind Offshore",
    "Solar",
    "Hydro Run-of-river and pondage",
]


def installed_capacity(raw_dir: Path) -> pd.DataFrame:
    """Installed MW by zone, technology and year, from the annual register."""
    rows = []
    for path in sorted((raw_dir / "capacity").glob("*.parquet")):
        stem = path.stem
        zone, _, year = stem.rpartition("_")
        df = pd.read_parquet(path)
        if df.empty:
            continue
        series = df.iloc[0]                      # one row per year
        for tech, mw in series.items():
            if pd.notna(mw):
                rows.append({"zone": zone, "year": int(year),
                             "tech": str(tech).split("|")[0].strip(),
                             "mw": float(mw)})
    return pd.DataFrame(rows)


def capacity_factors(generation: pd.DataFrame, capacity: pd.DataFrame,
                     techs: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """p_max_pu profiles, plus a report on how often they had to be clipped.

    Installed capacity is annual and generation is hourly, so each hour is
    divided by its own year's register. A factor above 1 means the register is
    behind the fleet - generation from units not yet listed - which is common
    for solar. Those are clipped, and the clip rate is reported rather than
    hidden: a technology clipping in 5% of hours has a capacity problem worth
    knowing about before it becomes a price problem.
    """
    techs = techs or RES_TECHS
    lookup = capacity.set_index(["zone", "year", "tech"])["mw"].to_dict()

    profiles, report = {}, []
    years = generation.index.year

    for col in generation.columns:
        zone, _, tech = col.partition("|")
        if tech not in techs:
            continue

        registered = pd.Series(
            [lookup.get((zone, y, tech)) for y in years], index=generation.index,
            dtype="float64",
        )
        observed_peak = float(generation[col].quantile(0.999))
        reg_max = float(registered.max()) if registered.notna().any() else 0.0

        if observed_peak <= 0:
            continue                       # nothing ever generated - not a plant

        # Third time this principle has been needed, so state it plainly:
        # where the published register disagrees with observed behaviour,
        # trust the observation. Switzerland reports solar and wind generation
        # with no capacity registered at all; its run-of-river generation
        # implies six times the registered MW; French offshore wind is
        # commissioning faster than the annual snapshot updates. A register
        # that agrees with the data is kept untouched.
        if reg_max <= 0:
            denom = pd.Series(observed_peak, index=generation.index)
            source = "observed (no register)"
        elif observed_peak > reg_max:
            denom = registered.clip(lower=observed_peak)
            source = "observed (register stale)"
        else:
            denom = registered
            source = "register"

        cf = (generation[col] / denom).clip(0, 1)
        n = int(generation[col].notna().sum())
        if not n:
            continue

        pct_over = (round(100 * float((generation[col] / registered > 1).sum()) / n, 2)
                    if reg_max > 0 else None)

        profiles[col] = cf
        report.append({
            "column": col,
            "hours": n,
            "mean_cf": round(float(cf.mean()), 4),
            "registered_mw": round(reg_max, 1),
            "observed_peak_mw": round(observed_peak, 1),
            "p_nom_source": source,
            "pct_above_1_vs_register": pct_over,
        })

    return pd.DataFrame(profiles, index=generation.index), pd.DataFrame(report)


def boundary_net_position(net_flows: pd.DataFrame, zones: list[str],
                          external: list[str]) -> pd.DataFrame:
    """Net export to the unmodelled world, per zone, MW.

    Positive means the zone exports across its boundary borders. In the model
    this becomes a fixed load (export) or generator (import), which is the
    v1 simplification: the outside world does not respond to anything.
    """
    out = {z: pd.Series(0.0, index=net_flows.index) for z in zones}
    for border in external:
        if border not in net_flows.columns:
            continue
        a, b = border.split(">")
        flow = net_flows[border].fillna(0)
        if a in out:
            out[a] = out[a] + flow          # a exports when positive
        if b in out:
            out[b] = out[b] - flow          # b imports when positive
    frame = pd.DataFrame(out)
    frame.columns.name = "zone"
    return frame


# ENTSO-E production types this model actually builds. Anything else that
# appears in the generation tables is real output with no representation here.
REPRESENTED = {
    "Nuclear", "Fossil Brown coal/Lignite", "Fossil Hard coal", "Fossil Gas",
    "Fossil Oil", "Biomass", "Waste",                      # the thermal fleet
    "Solar", "Wind Onshore", "Wind Offshore",              # capacity factors
    "Hydro Run-of-river and pondage", "Hydro Run-of-river and poundage",
    "Hydro Water Reservoir", "Hydro Pumped Storage",       # hydro profiles
}


def unmodelled_generation(load: pd.DataFrame, generation: pd.DataFrame,
                          flows: pd.DataFrame, zones: list[str],
                          threshold: float = 0.05,
                          shapes: dict | None = None,
                          bidding_share: dict | None = None):
    """Generation the MODEL is missing, per zone per hour.

    An earlier version of this measured the wrong thing. It computed the DATA's
    internal imbalance - load plus exports minus total published generation -
    which found 11.4% in the Netherlands and 16.7% in Switzerland. But the
    model is missing more than that: it is also missing every ENTSO-E
    production type it does not build.

    The Netherlands is the case that exposed it. Its register lists 1 MW under
    "Other"; its generation table reports an average of 4,393 MW there. Dutch
    load plus net exports averages 14,442 MW and the generation this model
    represents comes to 8,541, so the model is short 5,900 MW - 45% of Dutch
    load - in every hour. Supplying only the 1,507 MW data imbalance left the
    Dutch net position inverted by three gigawatts and the zone swinging
    between surplus and load shedding.

    The gap is therefore measured against what the model REPRESENTS, and it is
    supplied in two parts, because they have different shapes:

      A. Output published under types the model does not build. This is real
         hourly data, so it carries its own profile - mostly flat industrial
         and CHP output in the Dutch case.
      B. Whatever imbalance remains after that. Nobody published it, so it
         needs a shape: a borrowed one where the gap is known to be solar
         (see `shapes`), otherwise a typical-day profile by month and hour.

    Returns TWO frames, split by SHAPE rather than by whether the output was
    published:

      FLAT      the daily-minimum floor - industrial, CHP and waste-gas output
                that runs through the night and does not pay to run. Bids zero.
      VARIABLE  everything above that floor, plus whatever was never published
                at all. Behind-the-meter solar in the Dutch case, which loses
                its subsidy when curtailed and therefore does pay to run.

    Getting this split wrong in either direction is visible in the results.
    Giving all 5.9 GW a negative bid pushed Dutch negative hours to 1,200
    against an observed 458 and dragged Germany, Belgium and Poland with it;
    giving all of it a zero bid left the Netherlands at 297 with its rooftop
    PV unable to set a price it sets every sunny weekend in reality.

    Zones whose total correction is below `threshold` of load are left alone.
    Six of eight are, which is the point - this is a data repair for the two
    zones that need it, not a free parameter for all of them.
    """
    out, out_b = {}, {}
    for zone in zones:
        cols = [c for c in generation.columns if c.startswith(f"{zone}|")]
        if zone not in load or not cols:
            continue
        rep = [c for c in cols if c.split("|", 1)[1] in REPRESENTED]
        unrep = [c for c in cols if c not in rep]

        exports = pd.Series(0.0, index=flows.index)
        for border in flows.columns:
            a, _, b = border.partition(">")
            if a == zone:
                exports = exports + flows[border].fillna(0.0)
            elif b == zone:
                exports = exports - flows[border].fillna(0.0)

        idx = generation.index.intersection(load.index).intersection(flows.index)
        # A: published, unrepresented output - keep its own shape
        part_a = (generation[unrep].sum(axis=1).reindex(idx).fillna(0.0)
                  if unrep else pd.Series(0.0, index=idx))
        # B: what nobody published at all
        published = generation[cols].sum(axis=1).reindex(idx).fillna(0.0)
        part_b = (load[zone].reindex(idx) + exports.reindex(idx)
                  - published).clip(lower=0)

        total_mean = float((part_a + part_b).mean())
        if total_mean < threshold * float(load[zone].reindex(idx).mean()):
            continue

        shape_col = (shapes or {}).get(zone)
        if shape_col and shape_col in generation:
            shape = generation[shape_col].reindex(idx).fillna(0.0).clip(lower=0)
            key = [idx.year, idx.month]
            energy = part_b.groupby(key).transform("sum")
            weight = shape.groupby(key).transform("sum")
            shaped = (shape / weight.replace(0, pd.NA)).fillna(0.0) * energy
        else:
            # The real hourly gap, not a typical day. Averaging by month and
            # hour smooths off exactly the peaks that matter: Switzerland was
            # left 173 MW short in 22 hours of the year, shedding load at VOLL,
            # and those 22 hours accounted for essentially its entire price
            # standard deviation of 163 against an observed 39.
            #
            # This is a repair of published data, not a driver, so using the
            # observed hourly value is more accurate and no less honest than
            # inventing a smooth one.
            shaped = part_b

        # Split part A by SHAPE, not by whether it was published. Dutch
        # "Other" is a mixture: a flat 2,100 MW that runs through the night,
        # plus a solar-shaped bulge peaking near 5,600 MW at midday. Its
        # night/midday ratio is 0.285 against 0.00 for pure solar and 1.12 for
        # Dutch gas, and the midday bulge is 22% of the Dutch PV register -
        # the same fraction Germany's midday solar is of its own. It is
        # behind-the-meter rooftop PV, reported under "Other" because it is not
        # metered per unit.
        #
        # The two halves bid differently, which is the whole reason to separate
        # them: rooftop PV loses its subsidy when curtailed and so pays to run;
        # industrial and CHP output does not. The daily minimum is the floor -
        # a robust baseline that still follows seasonal drift in industrial
        # output - and everything above it is the diurnal component.
        floor = part_a.groupby(part_a.index.date).transform("min")
        flat = floor.clip(lower=0)
        diurnal = (part_a - floor).clip(lower=0)

        # A SHARE of the variable part bids; the rest does not. This is a
        # split in VOLUME, not a discount on price, and the distinction is not
        # cosmetic: whether an hour clears below zero depends on whether a
        # negative bidder is MARGINAL, not on how far below zero it offers.
        # Bidding the whole Dutch diurnal component at -4 instead of -8 gave
        # exactly the same 971 negative hours; moving half the megawatts to a
        # zero offer is what actually changes the count.
        share = float((bidding_share or {}).get(zone, 1.0))
        variable = (diurnal + shaped)

        out[zone] = (flat + variable * (1.0 - share)).reindex(load.index).fillna(0.0)
        out_b[zone] = (variable * share).reindex(load.index).fillna(0.0)
        log.info("%s: unmodelled %.0f MW mean (%.0f published as an "
                 "unrepresented type, %.0f never published)",
                 zone, total_mean, float(part_a.mean()), float(part_b.mean()))

    published = pd.DataFrame(out, index=load.index)
    unpublished = pd.DataFrame(out_b, index=load.index)
    published.columns.name = unpublished.columns.name = "zone"
    return published, unpublished
