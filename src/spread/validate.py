"""The stopping rule, in one place.

Six criteria were written into the design before package C began, expanded to
fifteen measurements across the two borders and three zones. They exist so that
calibration has an end: without them a fundamentals analyst will always find one
more thing to fix.

They live here rather than inside a script because two things need them - the
model run itself, and any sweep over parameters - and a rule that is
implemented twice will eventually be implemented differently.
"""
from __future__ import annotations

import pandas as pd

PAIRS = (("DE_LU", "FR"), ("DE_LU", "PL"))
LEVEL_ZONES = ("DE_LU", "FR", "PL")
DECILE_ZONES = ("DE_LU", "PL")
MEANINGFUL = 5.0        # EUR/MWh, for the supplementary sign test


def stopping_rule(price: pd.DataFrame, actual: pd.DataFrame) -> list[tuple]:
    """Return [(name, measured, passed), ...] for every criterion."""
    out = []

    for zone in LEVEL_ZONES:
        if zone in price and zone in actual:
            m, a = float(price[zone].mean()), float(actual[zone].mean())
            err = 100 * (m - a) / a if a else float("nan")
            out.append((f"{zone} mean price within 15%", f"{err:+.1f}%",
                        abs(err) <= 15))

    for x, y in PAIRS:
        if x not in price or y not in price:
            continue
        sim, act = price[x] - price[y], actual[x] - actual[y]
        both = sim.dropna().index.intersection(act.dropna().index)
        sim, act = sim.loc[both], act.loc[both]

        d = float(sim.mean() - act.mean())
        out.append((f"{x}-{y} mean spread within EUR 5", f"{d:+.2f}", abs(d) <= 5))

        ratio = float(sim.std() / act.std()) if act.std() else float("nan")
        out.append((f"{x}-{y} spread sd within 25%", f"{ratio:.2f}x",
                    0.75 <= ratio <= 1.25))

        pos, neg = act > 0, act < 0
        ap = float((sim[pos] > 0).mean()) if pos.any() else float("nan")
        an = float((sim[neg] < 0).mean()) if neg.any() else float("nan")
        out.append((f"{x}-{y} sign agreement, both directions",
                    f"{100 * ap:.0f}% / {100 * an:.0f}%", ap >= 0.70 and an >= 0.70))

        corr = float(sim.corr(act))
        out.append((f"{x}-{y} hourly spread correlation > 0.5", f"{corr:.2f}",
                    corr > 0.5))

    for zone in DECILE_ZONES:
        if zone not in price or zone not in actual:
            continue
        m, a = price[zone].dropna(), actual[zone].dropna()
        for q, name in ((0.90, "p90"), (0.10, "p10")):
            mv, av = float(m.quantile(q)), float(a.quantile(q))
            err = 100 * (mv - av) / abs(av) if av else float("nan")
            out.append((f"{zone} {name} within 25%", f"{mv:.0f} vs {av:.0f}",
                        abs(err) <= 25))
    return out


def supplementary(price: pd.DataFrame, actual: pd.DataFrame) -> list[tuple]:
    """Reported ALONGSIDE the rule, never in place of it.

    The sign-agreement criterion counts every hour equally, including hours
    where the true spread is under a euro and its sign is close to arbitrary.
    DE-PL sits within +/-5 EUR of zero for a large share of hours, so part of
    that test is scoring coin flips.

    Restricting it to hours where the actual spread exceeds MEANINGFUL says
    something the unrestricted number cannot: whether the model gets direction
    right when direction means something. It is NOT a replacement. The rule was
    tightened once already, after the first version turned out to be passable
    by predicting "negative, always"; loosening it now, knowing which change
    would make it pass, would be the same error with the sign reversed.
    """
    out = []
    for x, y in PAIRS:
        if x not in price or y not in price:
            continue
        sim, act = price[x] - price[y], actual[x] - actual[y]
        both = sim.dropna().index.intersection(act.dropna().index)
        sim, act = sim.loc[both], act.loc[both]
        pos, neg = act > MEANINGFUL, act < -MEANINGFUL
        ap = float((sim[pos] > 0).mean()) if pos.any() else float("nan")
        an = float((sim[neg] < 0).mean()) if neg.any() else float("nan")
        share = 100 * float((pos | neg).mean())
        out.append((f"{x}-{y} sign agreement where |spread| > EUR {MEANINGFUL:.0f}",
                    f"{100 * ap:.0f}% / {100 * an:.0f}%  (on {share:.0f}% of hours)"))
    return out


def congestion_timing(links_p0: pd.DataFrame, p_nom: dict,
                      observed: pd.DataFrame, tol: float = 0.98) -> pd.DataFrame:
    """Does the model congest a border in the SAME HOURS reality did?

    The spread criteria split oddly: distributions match - means within EUR 2,
    standard deviations at 0.87 and 1.04 - while hourly correlation sits at
    0.41 and 0.31. A sixfold change in the reserve requirement moved the
    correlations by 0.03, so price formation is not the cause.

    A spread is non-zero only when something binds, so the natural suspect is
    congestion TIMING. The model's capacity is smooth: a rolling weekly maximum
    on the links, static annual quantiles on the net positions. Within any week
    its congestion is therefore driven purely by price differences, while real
    allocated capacity moves hour to hour with the flow-based domain. That
    would give exactly what is observed - congesting about as OFTEN as reality
    and in different HOURS.

    The test is a phi coefficient on the binary "at capacity" indicator, which
    is just Pearson correlation of two 0/1 series. If the shares agree and phi
    is near zero, the diagnosis holds - and the conclusion is a limitation to
    document, not a bug to fix, because the CORE domain parameters that set
    hourly capacity are not published.

    SUPERSEDED, and the paragraph above is left standing because it was wrong
    in a specific way worth keeping. Two defects were being measured with one
    ruler, and the ruler was the wrong one:

      - it measures link p_nom, which net_position_limits() deliberately
        leaves slack, so it reports 0.1% and "never" for borders whose spreads
        the model is in fact producing;
      - its proxy for REAL congestion is flow within 2% of that border's own
        99th percentile. That measures flow SATURATION. Under flow-based
        coupling a border is constrained without its own flow being at any
        border-level maximum, so this understates real congestion badly - it
        put DE-PL at 1-5% where price separation puts it at 84.5%.

    Anything rejected on the strength of the 1-5% figure - per-border capacity
    limits, in particular - was rejected against a target that was wrong by an
    order of magnitude. See price_separation() below, which is the measurement
    that should have been made first.

    That does NOT reopen per-border limits, and the first draft of this note
    said it did. Price separation and flow saturation are different things:
    reality separates DE-FR in 78% of hours without the DE-FR interconnector
    being full in 78% of hours, because under flow-based coupling the binding
    element is a critical branch somewhere in the shared domain and its shadow
    price reaches the border as a price difference. Reproducing 78% separation
    with a static per-border limit would need that limit near the 22nd
    percentile of observed flow, which is what collapsed the flows and the
    prices the first time it was tried.

    And on the two SCORED borders the frequency gap is the smaller half of the
    problem - DE-PL 68.5% modelled against 84.5% observed, DE-FR 53.2% against
    78.4% - while the overlap sits at chance on both. The paragraph two above
    is therefore right about the scored borders for the reason it gives, and
    was reached with an instrument that could not have shown it.
    """
    rows = []
    for link in links_p0.columns:
        if link not in observed:
            continue
        cap = float(p_nom.get(link, 0.0))
        obs = observed[link].dropna()
        idx = links_p0.index.intersection(obs.index)
        if cap <= 0 or len(idx) < 100:
            continue

        mod_bind = (links_p0[link].reindex(idx).abs() >= tol * cap)
        obs_cap = float(obs.abs().quantile(0.99))
        obs_bind = (obs.reindex(idx).abs() >= tol * obs_cap)
        if not mod_bind.any() or not obs_bind.any():
            continue

        both = float((mod_bind & obs_bind).sum())
        rows.append({
            "border": link,
            "mod_%": round(100 * float(mod_bind.mean()), 1),
            "obs_%": round(100 * float(obs_bind.mean()), 1),
            "phi": round(float(mod_bind.astype(float).corr(
                obs_bind.astype(float))), 3),
            "hit_%": round(100 * both / max(float(obs_bind.sum()), 1), 1),
        })
    return pd.DataFrame(rows).set_index("border") if rows else pd.DataFrame()


def net_position_binding(links_p0: pd.DataFrame, links: pd.DataFrame,
                         bounds: pd.DataFrame, observed: pd.DataFrame | None = None,
                         tol: float = 0.99) -> pd.DataFrame:
    """How often does the NET POSITION bound bind, per zone?

    congestion_timing() measures link p_nom, and reported DE-PL binding in 0.1%
    of hours and DE-FR never - while the model still produced a DE-PL mean
    spread of -20 EUR/MWh. Uncongested links cannot do that. The reason is that
    p_nom is no longer the binding constraint: net_position_limits() moved the
    limit onto each zone's total exchange, so the links sit far below their
    thermal ratings while the zone hits its net-position bound. The old
    diagnostic was measuring a constraint that is deliberately slack.

    This measures the constraint that actually binds. It works from flows
    rather than duals because the chunked solve releases each chunk's model
    before the next one starts, so the shadow prices are gone by the end.
    """
    rows = []
    for zone in bounds.index:
        out = [k for k in links.index
               if k in links_p0.columns and links.at[k, "bus0"] == zone]
        inn = [k for k in links.index
               if k in links_p0.columns and links.at[k, "bus1"] == zone]
        if not out and not inn:
            continue
        net = (links_p0[out].sum(axis=1) if out else 0.0) \
            - (links_p0[inn].sum(axis=1) if inn else 0.0)
        hi, lo = float(bounds.at[zone, "hi_mw"]), float(bounds.at[zone, "lo_mw"])
        at_hi = net >= tol * hi
        at_lo = net <= tol * lo
        row = {
            "zone": zone,
            "hi_mw": round(hi), "lo_mw": round(lo),
            "mod_mean_mw": round(float(net.mean())),
            "at_export_%": round(100 * float(at_hi.mean()), 1),
            "at_import_%": round(100 * float(at_lo.mean()), 1),
            "bound_%": round(100 * float((at_hi | at_lo).mean()), 1),
        }
        if observed is not None and zone in observed:
            obs = observed[zone].reindex(net.index)
            row["obs_mean_mw"] = round(float(obs.mean()))
            row["obs_at_bound_%"] = round(
                100 * float(((obs >= tol * hi) | (obs <= tol * lo)).mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("zone") if rows else pd.DataFrame()


def price_separation(price: pd.DataFrame, actual: pd.DataFrame,
                     borders, eps: float = 0.5) -> pd.DataFrame:
    """Do the model's borders separate in the SAME HOURS reality's did?

    The definitive congestion test, and the one that does not care which
    constraint binds. Under implicit market coupling two zones clear at the
    SAME price whenever the border between them is not constrained, so
    |price difference| > eps IS congestion - observable directly, on both
    sides, with no NTC and no flow-based domain parameters needed.

    That matters here because the model's binding constraint (a net-position
    bound) and reality's (a flow-based domain) are different objects and cannot
    be compared like for like. Their PRICE consequences can.

    Reads as: does the model separate as OFTEN as reality (mod_% vs obs_%), and
    in the same HOURS (phi, hit_%)? A model that separates far less often has
    one price where reality has several, which caps every spread statistic that
    depends on hour-by-hour timing no matter how well the distributions match.
    """
    rows = []
    for x, y in borders:
        if x not in price or y not in price or x not in actual or y not in actual:
            continue
        sim = (price[x] - price[y]).dropna()
        act = (actual[x] - actual[y]).dropna()
        idx = sim.index.intersection(act.index)
        if len(idx) < 100:
            continue
        mod_sep = sim.loc[idx].abs() > eps
        obs_sep = act.loc[idx].abs() > eps
        both = float((mod_sep & obs_sep).sum())
        rows.append({
            "border": f"{x}-{y}",
            "mod_%": round(100 * float(mod_sep.mean()), 1),
            "obs_%": round(100 * float(obs_sep.mean()), 1),
            "phi": round(float(mod_sep.astype(float).corr(
                obs_sep.astype(float))), 3),
            "hit_%": round(100 * both / max(float(obs_sep.sum()), 1), 1),
            # If the model's separated hours were drawn at random, the hit rate
            # would equal mod_%. Printing the benchmark next to the measurement
            # stops it having to be re-derived every time the table is read.
            "if_random_%": round(100 * float(mod_sep.mean()), 1),
        })
    return pd.DataFrame(rows).set_index("border") if rows else pd.DataFrame()
