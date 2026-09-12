"""Is physical flow the same thing as commercial exchange? Per border.

    python scripts/16_commercial_vs_physical.py

Why this matters here
---------------------
Every border capacity and every net-position bound in this model was derived
from PHYSICAL flow, because that is what was pulled first. Physical flow is
what crossed the wire. The day-ahead auction allocates something else:
scheduled commercial exchange, the power actually traded across that border.

The two differ by loop flow - power sold between two points inside Germany
that physically routes through Poland and Czechia because that is where the
impedance sends it. It occupies transmission the market coupling never had to
allocate, and it is historically largest on exactly the German-Polish and
German-Czech borders, which is why both countries installed phase shifters.

If the model sized DE-PL from physical flow, it handed the optimiser capacity
the auction never had, and the two zones could never separate. That would show
up as what we currently see: each zone individually well calibrated, and the
spread between them with a third of the variance it should have.

This script does not fix anything. It measures whether the hypothesis is true
before any input is changed.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402

from spread.config import PROCESSED, RAW, load_config        # noqa: E402
from spread.process import build_commercial_flows            # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("commercial")


def main() -> None:
    cfg = load_config()
    borders = cfg["borders_internal"]
    index = pd.date_range(
        pd.Timestamp(cfg["period"]["start"], tz="UTC"),
        pd.Timestamp(cfg["period"]["end"], tz="UTC"),
        freq="h", inclusive="left", name="timestamp",
    )

    log.info("assembling scheduled commercial exchange ...")
    comm = build_commercial_flows(RAW, index, borders)
    comm.to_parquet(PROCESSED / "commercial_flows.parquet")
    log.info("  wrote commercial_flows.parquet  %d borders", comm.shape[1])

    phys = pd.read_parquet(PROCESSED / "net_flows.parquet").reindex(index)

    rows = []
    for border in borders:
        if border not in comm or border not in phys:
            continue
        c, f = comm[border].dropna(), phys[border].dropna()
        both = c.index.intersection(f.index)
        if len(both) < 100:
            continue
        c, f = c.loc[both], f.loc[both]
        rows.append({
            "border": border,
            "comm_abs_mw": round(float(c.abs().mean())),
            "phys_abs_mw": round(float(f.abs().mean())),
            "ratio": round(float(c.abs().mean() / f.abs().mean()), 2)
            if f.abs().mean() else float("nan"),
            "comm_p99": round(float(c.abs().quantile(0.99))),
            "phys_p99": round(float(f.abs().quantile(0.99))),
            "loop_mw": round(float((f - c).abs().mean())),
            "corr": round(float(c.corr(f)), 2),
            "hours": len(both),
        })

    out = pd.DataFrame(rows).set_index("border")
    print("\n" + "=" * 92)
    print("SCHEDULED COMMERCIAL EXCHANGE vs PHYSICAL FLOW, per border")
    print("=" * 92)
    print(out.sort_values("ratio").to_string())
    print("\n  ratio  = commercial / physical, on mean absolute flow")
    print("  loop_mw = mean |physical - commercial|, the unscheduled component")
    print("\n  A ratio near 1 means the two series say the same thing and the")
    print("  model's capacities are fine. A ratio well below 1 means the auction")
    print("  traded far less across that border than crossed it, and sizing the")
    print("  link from physical flow over-coupled those two zones.")

    print("\n" + "=" * 92)
    print("WHAT THIS WOULD CHANGE: link p_nom at the 99th percentile")
    print("=" * 92)
    caps = pd.read_csv(PROCESSED / "link_capacity.csv", index_col=0)
    rows = []
    for border in out.index:
        if border not in caps.index:
            continue
        current = max(float(caps.at[border, "p_nom_fwd_mw"]),
                      float(caps.at[border, "p_nom_rev_mw"]))
        rows.append({
            "border": border,
            "p_nom_now_mw": round(current),
            "p_nom_commercial_mw": round(float(out.at[border, "comm_p99"])),
            "change_%": round(100 * (out.at[border, "comm_p99"] / current - 1), 1)
            if current else float("nan"),
        })
    print(pd.DataFrame(rows).set_index("border").to_string())


if __name__ == "__main__":
    main()
