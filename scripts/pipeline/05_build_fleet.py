"""Build the thermal fleet table. Run in .venv-fleet, not the model venv.

    py -3.14 -m venv .venv-fleet
    .venv-fleet\\Scripts\\activate
    pip install -r requirements-fleet.txt
    python scripts\\05_build_fleet.py
    deactivate

powerplantmatching pins pandas<3 while PyPSA needs pandas>=3, so the two
cannot share an environment. This runs once and commits its output to
data/fleet/fleet.csv, after which the model environment never needs it.

Output columns: zone, tech, name, capacity_mw, commissioned, efficiency,
co2_t_per_mwh_th, vom_eur_per_mwh
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import powerplantmatching as pm
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "fleet"

TARGET_YEAR = 2025      # the calibration year: fleet as it stood then
MIN_MW = 10.0           # below this, units are engines rather than plant

ENTSOE_FUEL = {
    "Fossil Brown coal/Lignite": "Lignite",
    "Fossil Hard coal": "Hard Coal",
    "Fossil Gas": "Natural Gas",
    "Fossil Oil": "Oil",
    "Nuclear": "Nuclear",
    "Waste": "Waste",
    "Biomass": "Bioenergy",
}
FUEL_TO_TECH = {
    "Lignite": "Lignite", "Hard Coal": "Hard Coal", "Natural Gas": "CCGT",
    "Oil": "Oil", "Nuclear": "Nuclear", "Waste": "Waste",
    "Bioenergy": "Bioenergy",
}

# powerplantmatching uses country names; the model uses bidding zones.
# Luxembourg joins Germany because DE-LU is a single zone.
COUNTRY_TO_ZONE = {
    "Germany": "DE_LU", "Luxembourg": "DE_LU", "France": "FR", "Poland": "PL",
    "Netherlands": "NL", "Belgium": "BE", "Austria": "AT",
    "Switzerland": "CH", "Czech Republic": "CZ", "Czechia": "CZ",
}

# Map (Fueltype, Technology) onto the technology names used in technology.yaml.
def classify(fueltype: str, technology: str) -> str | None:
    ft, tech = str(fueltype), str(technology)
    if ft == "Lignite":
        return "Lignite"
    if ft == "Hard Coal":
        return "Hard Coal"
    if ft == "Natural Gas":
        if "CCGT" in tech:
            return "CCGT"
        if "OCGT" in tech or "Gas Turbine" in tech:
            return "OCGT"
        return "Steam Turbine"
    if ft == "Oil":
        return "Oil"
    if ft == "Nuclear":
        return "Nuclear"
    if ft == "Bioenergy":
        return "Bioenergy"
    if ft == "Waste":
        return "Waste"
    return None                      # hydro, wind, solar handled as profiles


def efficiency_for(tech: str, year: float, bands: dict,
                   name: str | None = None) -> float | None:
    """Efficiency interpolated continuously across the vintage bands.

    Stepping between four discrete band values gave every 2005-2015 CCGT
    exactly 0.57, which put a 15 GW flat step in the German merit order:
    demand moved along it without moving the price, and simulated prices came
    out with a standard deviation of 3 EUR/MWh over a week. Efficiency
    improves continuously with vintage, so interpolate.

    A small deterministic spread on top breaks remaining ties between units of
    the same year - real plants of one vintage differ by more than nothing,
    and the alternative is a staircase with artificial flat treads.
    """
    table = bands.get(tech)
    if not table:
        return None
    if pd.isna(year):
        year = 1995                  # unknown vintage: assume mid-fleet

    anchors = sorted(table)
    values = [table[a] for a in anchors]
    if year <= anchors[0]:
        eff = values[0]
    elif year >= anchors[-1]:
        eff = values[-1]
    else:
        eff = float(np.interp(year, anchors, values))

    if name:
        # deterministic, reproducible, +/- 1.5% of the interpolated value
        jitter = ((hash(name) % 1000) / 1000 - 0.5) * 0.03
        eff *= 1 + jitter
    return round(float(eff), 4)


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config" / "technology.yaml", encoding="utf-8"))
    zones = yaml.safe_load(open(ROOT / "config" / "zones.yaml", encoding="utf-8"))["zones"]

    print("fetching powerplantmatching dataset (first run downloads and caches) ...")
    df = pm.powerplants()
    print(f"  {len(df)} plants across Europe")

    df["zone"] = df["Country"].map(COUNTRY_TO_ZONE)
    df = df[df["zone"].isin(zones)].copy()
    print(f"  {len(df)} in the eight modelled zones")

    df["tech"] = [classify(f, t) for f, t in zip(df["Fueltype"], df["Technology"])]
    df = df[df["tech"].notna()].copy()
    print(f"  {len(df)} thermal units after dropping hydro, wind and solar")

    # powerplantmatching returns the full HISTORICAL database. Without this
    # filter Germany carries 27 GW of nuclear it shut in April 2023, and coal
    # fleets that closed years ago. DateOut is the retirement year.
    if "DateOut" in df.columns:
        out_year = pd.to_numeric(df["DateOut"], errors="coerce")
        df = df[out_year.isna() | (out_year >= TARGET_YEAR)].copy()
        print(f"  {len(df)} still operating in {TARGET_YEAR}")
    if "DateIn" in df.columns:
        in_year = pd.to_numeric(df["DateIn"], errors="coerce")
        df = df[in_year.isna() | (in_year <= TARGET_YEAR)].copy()
        print(f"  {len(df)} commissioned by {TARGET_YEAR}")

    # The German register lists every standby diesel and biogas engine. Those
    # are not price-setting plant; keep the tail's total so the loss is known.
    small = df[df["Capacity"] < MIN_MW]
    print(f"  dropping {len(small)} units below {MIN_MW} MW "
          f"({small['Capacity'].sum() / 1000:.1f} GW in total)")
    df = df[df["Capacity"] >= MIN_MW].copy()
    print(f"  {len(df)} units in the fleet")

    df["efficiency"] = [
        efficiency_for(t, y, cfg["efficiency"], str(nm))
        for t, y, nm in zip(df["tech"],
                            df.get("DateIn", pd.Series(index=df.index)),
                            df.get("Name", df.index.astype(str)))
    ]

    # powerplantmatching sometimes carries a plant-specific efficiency; prefer
    # it where it exists and is credible, otherwise the vintage band.
    if "Efficiency" in df.columns:
        own = pd.to_numeric(df["Efficiency"], errors="coerce")
        band = pd.to_numeric(df["efficiency"], errors="coerce")
        # Accept a plant-specific value only where it refines the vintage band
        # rather than contradicting it. Without this the data carries 24% CCGTs
        # and 61% steam turbines, which are not plants but bad records.
        credible = own.notna() & band.notna() & (own - band).abs().le(0.08)
        df.loc[credible, "efficiency"] = own[credible]
        print(f"  {int(credible.sum())} units refined by a plant-specific efficiency")

    fuel_of = {"Lignite": "Lignite", "Hard Coal": "Hard Coal",
               "CCGT": "Natural Gas", "OCGT": "Natural Gas",
               "Steam Turbine": "Natural Gas", "Oil": "Oil",
               "Nuclear": "Nuclear", "Bioenergy": "Bioenergy", "Waste": "Waste"}
    df["fuel"] = df["tech"].map(fuel_of)
    df["co2_t_per_mwh_th"] = df["fuel"].map(cfg["emission_factors"]).fillna(0.0)
    df["vom_eur_per_mwh"] = df["tech"].map(cfg["vom"]).fillna(3.0)

    out = (df.rename(columns={"Name": "name", "Capacity": "capacity_mw",
                              "DateIn": "commissioned"})
             [["zone", "tech", "fuel", "name", "capacity_mw", "commissioned",
               "efficiency", "co2_t_per_mwh_th", "vom_eur_per_mwh"]]
             .sort_values(["zone", "tech", "capacity_mw"],
                          ascending=[True, True, False])
             .reset_index(drop=True))

    # ---- reconcile totals against the ENTSO-E register ----
    #
    # Two sources, each good at a different thing. ENTSO-E is TSO-reported and
    # current, so it is authoritative on HOW MUCH capacity exists per zone and
    # fuel. powerplantmatching is a merge of historical registers with patchy
    # retirement dates - Belgium still carries reactors that closed - but it is
    # the only source for HOW that capacity is spread across vintages, which is
    # what sets the shape of the merit order.
    #
    # So: take the totals from ENTSO-E, the distribution from powerplantmatching,
    # and scale each group to match. Where a fuel exists in the register but has
    # no matched units - German biomass is 8.9 GW of plants mostly under 10 MW -
    # add one aggregate block at the technology's default efficiency.
    reg_path = ROOT / "data" / "processed" / "installed_capacity.csv"
    if reg_path.exists():
        reg = pd.read_csv(reg_path)
        # The register's LAST year is not the study's year. Belgium's nuclear
        # register reads 3,929 MW in 2024 and 2025 and 2,056 in 2026, when Doel 1,
        # Doel 2 and Tihange 1 come off - so reconciling to the maximum year scaled
        # every Belgian reactor down to fit a post-closure total and then applied it
        # to years those units were running. Doel 4 came out at 547 MW against a
        # real 1,039. The model was 1.8 GW short of Belgian must-run baseload in
        # every hour of 2024 and 2025, which is most of why Belgium produced 106
        # negative price hours against an observed 404.
        reg = reg[reg["year"] == min(reg["year"].unique(),
                                       key=lambda y: abs(y - TARGET_YEAR))]
        reg["fuel"] = reg["tech"].map(ENTSOE_FUEL)
        target = (reg[reg["fuel"].notna()]
                  .groupby(["zone", "fuel"])["mw"].sum())

        scaled, added = [], []
        for (zone, fuel), target_mw in target.items():
            if zone not in zones:
                continue
            grp = out[(out["zone"] == zone) & (out["fuel"] == fuel)]
            have = float(grp["capacity_mw"].sum())

            if target_mw <= 0:
                continue                                   # register says none
            if have <= 0:
                tech = FUEL_TO_TECH[fuel]
                added.append({
                    "zone": zone, "tech": tech, "fuel": fuel,
                    "name": f"{zone} {fuel} aggregate",
                    "capacity_mw": float(target_mw), "commissioned": pd.NA,
                    "efficiency": efficiency_for(tech, 2000, cfg["efficiency"]),
                    "co2_t_per_mwh_th": cfg["emission_factors"].get(fuel, 0.0),
                    "vom_eur_per_mwh": cfg["vom"].get(tech, 3.0),
                })
            else:
                factor = float(target_mw) / have
                out.loc[grp.index, "capacity_mw"] *= factor
                if abs(factor - 1) > 0.05:
                    scaled.append((zone, fuel, round(factor, 2)))

        # anything the register does not list at all is not in the fleet
        listed = set(target[target > 0].index)
        keep = [i for i in out.index
                if (out.at[i, "zone"], out.at[i, "fuel"]) in listed]
        dropped = len(out) - len(keep)
        out = out.loc[keep]

        if added:
            out = pd.concat([out, pd.DataFrame(added)], ignore_index=True)

        print(f"\n  scaled {len(scaled)} zone/fuel groups to the register")
        for z, f, k in sorted(scaled, key=lambda r: -abs(r[2] - 1))[:8]:
            print(f"      {z:6s} {f:<12s} x{k}")
        print(f"  added {len(added)} aggregate blocks where no units matched")
        for a in added:
            print(f"      {a['zone']:6s} {a['fuel']:<12s} {a['capacity_mw']/1000:.1f} GW")
        print(f"  dropped {dropped} units in fuels the register does not list")

    out = out.sort_values(["zone", "tech", "capacity_mw"],
                          ascending=[True, True, False]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "fleet.csv", index=False)
    print(f"\nwrote {OUT / 'fleet.csv'}  ({len(out)} units)")

    # ---- cross-check against the ENTSO-E register ----
    print("\n" + "=" * 74)
    print("FLEET CAPACITY, GW")
    print("=" * 74)
    pivot = (out.pivot_table(index="tech", columns="zone",
                             values="capacity_mw", aggfunc="sum")
             .div(1000).round(1).fillna(0))
    print(pivot.to_string())

    # The cross-check, actually performed. Writing one and not running it is
    # how 27 GW of decommissioned German nuclear got into the first version
    # of this table without anything complaining.
    reg_path = ROOT / "data" / "processed" / "installed_capacity.csv"
    if reg_path.exists():
        reg = pd.read_csv(reg_path)
        # The register's LAST year is not the study's year. Belgium's nuclear
        # register reads 3,929 MW in 2024 and 2025 and 2,056 in 2026, when Doel 1,
        # Doel 2 and Tihange 1 come off - so reconciling to the maximum year scaled
        # every Belgian reactor down to fit a post-closure total and then applied it
        # to years those units were running. Doel 4 came out at 547 MW against a
        # real 1,039. The model was 1.8 GW short of Belgian must-run baseload in
        # every hour of 2024 and 2025, which is most of why Belgium produced 106
        # negative price hours against an observed 404.
        reg = reg[reg["year"] == min(reg["year"].unique(),
                                       key=lambda y: abs(y - TARGET_YEAR))]
        reg["fuel"] = reg["tech"].map(ENTSOE_FUEL)
        reg = reg[reg["fuel"].notna()]

        theirs = reg.groupby(["zone", "fuel"])["mw"].sum().div(1000)
        ours = out.groupby(["zone", "fuel"])["capacity_mw"].sum().div(1000)
        cmp = pd.DataFrame({"fleet_gw": ours, "entsoe_gw": theirs}).fillna(0)
        cmp["ratio"] = (cmp["fleet_gw"] / cmp["entsoe_gw"].replace(0, float("nan"))
                        ).round(2)
        cmp = cmp.round(1)

        print("\n" + "=" * 74)
        print("CROSS-CHECK vs ENTSO-E REGISTER   (ratio far from 1.0 = a problem)")
        print("=" * 74)
        material = (cmp["fleet_gw"] > 0.2) | (cmp["entsoe_gw"] > 0.2)
        suspect = cmp[material & ((cmp["ratio"] > 1.25) | (cmp["ratio"] < 0.75)
                                  | cmp["ratio"].isna())]
        if suspect.empty:
            print("  every zone/fuel within 25% of the register")
        else:
            print(suspect.to_string())
            print(f"\n  {len(suspect)} of {len(cmp)} zone/fuel pairs disagree by "
                  f"more than 25%. Investigate before building on this.")
    else:
        print("\n  (run scripts/03_build_inputs.py first for the ENTSO-E cross-check)")

    print("\n" + "=" * 74)
    print("EFFICIENCY SPREAD WITHIN EACH TECHNOLOGY")
    print("=" * 74)
    spread = (out.groupby("tech")["efficiency"]
              .agg(["count", "min", "mean", "max"]).round(3))
    print(spread.to_string())

    missing = out["commissioned"].isna().sum()
    print(f"\n{missing} of {len(out)} units have no commissioning year "
          f"({100 * missing / len(out):.1f}%) - those took the 1995 default band.")


if __name__ == "__main__":
    main()
