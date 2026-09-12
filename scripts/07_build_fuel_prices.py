"""Build daily fuel and carbon price series.

    python scripts/07_build_fuel_prices.py

Reads whatever CSVs are in data/raw/fuel/ (see the README there for sources),
falls back to config/fuel_fallback.yaml for anything missing, and says loudly
which series are still synthetic.

Writes data/processed/fuel_prices.parquet with a daily UTC index:
    gas_eur_mwh_th, coal_eur_mwh_th, co2_eur_t
plus a source column per series so provenance travels with the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                       # noqa: E402
import yaml                                               # noqa: E402

from spread.config import PROCESSED, RAW, load_config     # noqa: E402

FUEL_DIR = RAW / "fuel"

# API2 is 6,000 kcal/kg NAR. 1 kcal = 1.163e-6 MWh, so a tonne carries
# 6e6 kcal x 1.163e-6 = 6.978 MWh of thermal energy.
MWH_TH_PER_TONNE_COAL = 6.978

DATE_HINTS = ("date", "day", "time", "datum", "period", "delivery")
PRICE_HINTS = ("price", "settle", "close", "clearing", "value", "eur", "usd")


def _find_column(df: pd.DataFrame, hints: tuple[str, ...],
                 exclude: str | None = None) -> str | None:
    """Locate a column by fuzzy name match, so most exports work unedited."""
    for hint in hints:
        for col in df.columns:
            name = str(col).lower()
            if hint in name and (exclude is None or exclude not in name):
                return col
    return None


def read_series(filename: str, label: str) -> tuple[pd.Series | None, str]:
    """Read one price CSV into a daily series, or return None with a reason."""
    path = FUEL_DIR / filename
    if not path.exists():
        return None, "no file"

    for kwargs in ({}, {"sep": ";"}, {"sep": "\t"}, {"skiprows": 1}):
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 2:
                break
        except Exception:
            continue
    else:
        return None, "could not parse"

    date_col = _find_column(df, DATE_HINTS)
    price_col = _find_column(df, PRICE_HINTS, exclude="date")
    if date_col is None or price_col is None:
        return None, f"columns not found in {list(df.columns)[:6]}"

    s = pd.Series(
        pd.to_numeric(df[price_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col], errors="coerce", utc=True, format="mixed"),
        name=label,
    ).dropna()
    s = s[~s.index.isna()]
    if s.empty:
        return None, "no usable rows"
    return s[~s.index.duplicated(keep="last")].sort_index(), f"{path.name} ({len(s)} rows)"


def fallback_series(fb: dict, key: str, index: pd.DatetimeIndex) -> pd.Series:
    monthly = {pd.Timestamp(m, tz="UTC"): v[key] for m, v in fb["monthly"].items()}
    s = pd.Series(monthly).sort_index()
    return s.reindex(index.union(s.index)).ffill().bfill().reindex(index)


def main() -> None:
    cfg = load_config()
    fb = yaml.safe_load(open(ROOT / "config" / "fuel_fallback.yaml", encoding="utf-8"))

    index = pd.date_range(
        pd.Timestamp(cfg["period"]["start"], tz="UTC"),
        pd.Timestamp(cfg["period"]["end"], tz="UTC"),
        freq="D", inclusive="left", name="date",
    )

    sources, out = {}, {}

    # ---- carbon ----
    eua, why = read_series("eua.csv", "co2_eur_t")
    if eua is not None:
        out["co2_eur_t"] = eua.reindex(index.union(eua.index)).ffill().reindex(index)
        sources["co2_eur_t"] = f"EEX auction: {why}"
    else:
        out["co2_eur_t"] = fallback_series(fb, "eua", index)
        sources["co2_eur_t"] = f"FALLBACK ({why})"

    # ---- gas ----
    ttf, why = read_series("ttf.csv", "gas_eur_mwh_th")
    if ttf is not None:
        out["gas_eur_mwh_th"] = ttf.reindex(index.union(ttf.index)).ffill().reindex(index)
        sources["gas_eur_mwh_th"] = f"TTF: {why}"
    else:
        out["gas_eur_mwh_th"] = fallback_series(fb, "ttf", index)
        sources["gas_eur_mwh_th"] = f"FALLBACK ({why})"

    # ---- coal, USD/t -> EUR/MWh_th ----
    fx_series, fx_why = read_series("fx.csv", "fx")
    if fx_series is not None:
        fx = fx_series.reindex(index.union(fx_series.index)).ffill().reindex(index)
    else:
        fx = pd.Series(fb["fx_usd_per_eur"], index=index)
        fx_why = f"constant {fb['fx_usd_per_eur']}"

    coal, why = read_series("coal.csv", "coal_usd_t")
    if coal is not None:
        usd_t = coal.reindex(index.union(coal.index)).ffill().reindex(index)
        sources["coal_eur_mwh_th"] = f"API2: {why}, FX {fx_why}"
    else:
        usd_t = fallback_series(fb, "api2", index)
        sources["coal_eur_mwh_th"] = f"FALLBACK ({why})"
    out["coal_eur_mwh_th"] = usd_t / MWH_TH_PER_TONNE_COAL / fx

    prices = pd.DataFrame(out, index=index).ffill().bfill()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PROCESSED / "fuel_prices.parquet")
    pd.Series(sources).to_csv(PROCESSED / "fuel_price_sources.csv",
                              header=["source"])

    # ---- report ----
    print("\n" + "=" * 72)
    print("SOURCES")
    print("=" * 72)
    synthetic = []
    for k, v in sources.items():
        flag = "  <-- SYNTHETIC" if v.startswith("FALLBACK") else ""
        print(f"  {k:<18} {v}{flag}")
        if v.startswith("FALLBACK"):
            synthetic.append(k)

    print("\n" + "=" * 72)
    print("ANNUAL MEANS")
    print("=" * 72)
    print(prices.groupby(prices.index.year).mean().round(2).to_string())

    # ---- the number this project turns on ----
    print("\n" + "=" * 72)
    print("IMPLIED COAL-GAS SWITCHING CARBON PRICE, EUR/t")
    print("=" * 72)
    print("  the CO2 price at which a 55% CCGT undercuts a 40% lignite unit,")
    print("  and separately a 42% hard coal unit\n")
    tech = yaml.safe_load(open(ROOT / "config" / "technology.yaml", encoding="utf-8"))
    ef = tech["emission_factors"]
    lig_fuel = tech["fixed_fuel_cost"]["Lignite"]

    def switch(gas, other_fuel, eff_gas, eff_other, f_other, tech_other):
        """Carbon price at which a CCGT undercuts the named technology."""
        gas_cost = gas / eff_gas + tech["vom"]["CCGT"]
        other_cost = other_fuel / eff_other + tech["vom"][tech_other]
        den = (ef[f_other] / eff_other) - (ef["Natural Gas"] / eff_gas)
        return (gas_cost - other_cost) / den if den else float("nan")

    yearly = prices.groupby(prices.index.year).mean()
    rows = []
    for year, r in yearly.iterrows():
        rows.append({
            "year": year,
            "gas": round(r["gas_eur_mwh_th"], 1),
            "coal": round(r["coal_eur_mwh_th"], 1),
            "actual_co2": round(r["co2_eur_t"], 1),
            "switch_vs_lignite": round(
                switch(r["gas_eur_mwh_th"], lig_fuel, 0.55, 0.40,
                       "Lignite", "Lignite"), 1),
            "switch_vs_hardcoal": round(
                switch(r["gas_eur_mwh_th"], r["coal_eur_mwh_th"], 0.55, 0.42,
                       "Hard Coal", "Hard Coal"), 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  actual_co2 above switch_vs_X means gas beat that fuel on average.")

    if synthetic:
        print("\n" + "!" * 72)
        print(f"  {len(synthetic)} series still synthetic: {', '.join(synthetic)}")
        print("  Fine for building the model. Replace before publishing anything.")
        print("!" * 72)


if __name__ == "__main__":
    main()
