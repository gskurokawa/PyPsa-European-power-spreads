"""Is the Dutch "Other" series solar, or is it industrial?

    python scripts/18_what_is_other.py

The Netherlands publishes 4,393 MW of average generation under the ENTSO-E
production type "Other" against 58 MW under "Solar" and 1 MW of registered
"Other" capacity. Something is being reported in the wrong place, and which
thing it is decides how that generation should BID: rooftop PV pays to run
because it loses its subsidy when curtailed, industrial and CHP output does
not.

Shape settles it. Solar is zero at night.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402

from spread.config import PROCESSED                          # noqa: E402

pd.set_option("display.width", 200)


def main() -> None:
    gen = pd.read_parquet(PROCESSED / "generation.parquet")
    gen = gen[gen.index.year == 2024]

    cols = ["NL|Other", "NL|Solar", "BE|Solar", "DE_LU|Solar",
            "NL|Fossil Gas", "NL|Wind Offshore"]
    cols = [c for c in cols if c in gen]

    print("=" * 96)
    print("MEAN OUTPUT BY HOUR OF DAY, MW   (UTC) - solar is zero at night")
    print("=" * 96)
    by_hour = gen[cols].groupby(gen.index.hour).mean().round(0)
    by_hour.index.name = "hour"
    print(by_hour.to_string())

    print("\n" + "=" * 96)
    print("NIGHT vs MIDDAY, and the ratio that gives it away")
    print("=" * 96)
    night = gen[cols].loc[(gen.index.hour <= 2) | (gen.index.hour >= 22)].mean()
    midday = gen[cols].loc[(gen.index.hour >= 10) & (gen.index.hour <= 13)].mean()
    summary = pd.DataFrame({
        "night_mw": night.round(0),
        "midday_mw": midday.round(0),
        "night/midday": (night / midday.replace(0, pd.NA)).round(3),
        "annual_mean": gen[cols].mean().round(0),
        "max_mw": gen[cols].max().round(0),
    })
    print(summary.to_string())
    print("\n  night/midday near 0.00 = solar.  near 1.00 = something that runs")
    print("  through the night: industrial, CHP, or waste-gas.")

    print("\n" + "=" * 96)
    print("MEAN BY MONTH, MW   - solar triples between December and June")
    print("=" * 96)
    by_month = gen[cols].groupby(gen.index.month).mean().round(0)
    by_month.index.name = "month"
    print(by_month.to_string())

    if "NL|Other" in gen:
        s = gen["NL|Other"]
        summer = s[s.index.month.isin([5, 6, 7])].mean()
        winter = s[s.index.month.isin([11, 12, 1])].mean()
        print(f"\n  NL|Other  summer {summer:,.0f} MW   winter {winter:,.0f} MW   "
              f"ratio {summer / winter:.2f}")
        print("  Dutch solar output runs about 3x higher in summer than winter.")


if __name__ == "__main__":
    main()
