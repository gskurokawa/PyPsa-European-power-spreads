"""Pull the observed series the flow-based constraints need but the model
does not solve for.

    python scripts/30_fetch_fixed_terms.py --test     # one request, checks the token
    python scripts/30_fetch_fixed_terms.py            # everything, restartable

Script 29 sized every term that has to move to the right-hand side of

    SUM over MODELLED zones of PTDF x NP  <=  RAM  -  SUM over FIXED terms

and found that only six of seventeen are worth collecting.  The nine external
virtual hubs - NorNed, SwePol, COBRA, the Danish and Nordic big hubs - have a
PTDF of exactly zero on every Core CNEC in the data, in 100% of rows.  They
contribute nothing and are dropped, with that measurement as the justification
rather than an assumption.

What remains, with its typical contribution in MW of RAM (average RAM 882):

    SI 182   SK 179   HR 123   HU 106   RO 62      the omitted Core zones
    ALEGrO   93 / 65                               the Belgium-Germany HVDC

ALEGRO
------
ALEGrO is the only direct interconnector between Belgium and Germany - they
have no AC connection - so the day-ahead scheduled exchange on that border is
its flow.  The domain represents it as two virtual hubs, hub_ALBE and
hub_ALDE, one per end.  Which sign goes with which end is a convention this
script does NOT guess: it stores the flow once, and script 31 settles the
orientation by testing which assignment reproduces the constraints JAO
reports as binding.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402
from entsoe import EntsoePandasClient                        # noqa: E402

from spread.config import RAW, api_key, load_config          # noqa: E402
from spread.entsoe_pull import pull, redact                  # noqa: E402

log = logging.getLogger("fixed")

# ALL TWELVE Core bidding zones, not just the five this model omits.
#
# Script 32 found the right constraint formulation - published RAM against a
# CORE net position - but could only reach a 22% hit rate, because the Core
# net positions built from this repo's own thirteen borders are incomplete for
# Austria, Czechia and Poland: their borders with Hungary, Slovakia, Slovenia
# and Croatia are missing, and those are gigawatt-scale.  Pulling the
# published net position for every Core zone sidesteps rebuilding them from
# borders altogether, and gives a series defined the same way JAO defines it.
CORE_ZONES = ["AT", "BE", "CZ", "DE_LU", "FR", "HR", "HU",
              "NL", "PL", "RO", "SI", "SK"]
OMITTED_CORE = CORE_ZONES
# ALEGrO: the one BE-DE link, both directions
ALEGRO_BORDERS = ["BE>DE_LU", "DE_LU>BE"]


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(ROOT / "logs" / "fixed_terms.log",
                                      encoding="utf-8")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end",   default="2025-01-01")
    ap.add_argument("--test",  action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--wait", action="store_true",
                    help="poll until the ENTSO-E platform answers, then pull. "
                         "Leave it running; it costs nothing while waiting.")
    ap.add_argument("--poll-minutes", type=int, default=180,
                    help="how often to probe while the platform is down. "
                         "Three hours is deliberately unhurried: a platform-"
                         "wide outage is not resolved by asking more often, "
                         "and a probe every ten minutes through the night is "
                         "just noise on someone else's service.")
    args = ap.parse_args()
    setup_logging()

    client = EntsoePandasClient(api_key=api_key())

    if args.wait:
        # Neither the cloud sandbox nor the desktop bridge can reach ENTSO-E,
        # so this has to wait here rather than being scheduled elsewhere.
        # A cheap probe, repeated, until the platform is answering; then the
        # real pull runs and this exits.
        s0 = pd.Timestamp("2024-01-01", tz="UTC")
        e0 = pd.Timestamp("2024-01-02", tz="UTC")
        waited = 0
        while True:
            try:
                client.query_load("SI", start=s0, end=e0)
                print(f"\n  platform answering after {waited} min - pulling now\n")
                break
            except Exception as exc:                          # noqa: BLE001
                print(f"  {time.strftime('%H:%M')}  still down after "
                      f"{waited} min: {redact(exc)[:70]}")
                time.sleep(args.poll_minutes * 60)
                waited += args.poll_minutes

    if args.test:
        s = pd.Timestamp("2024-01-01", tz="UTC")
        e = pd.Timestamp("2024-01-03", tz="UTC")

        # Is the platform unwell, or is this query type simply not served?
        # A 503 on everything is the first; a 503 on net position while load
        # works is the second, and means falling back to scheduled exchanges.
        print("  probe 1: an ordinary query, to see if the platform is up")
        for attempt in range(4):
            try:
                d = client.query_load("SI", start=s, end=e)
                print(f"    load(SI) OK - {len(d)} rows. Platform is up.")
                break
            except Exception as exc:                          # noqa: BLE001
                print(f"    attempt {attempt+1}: {redact(exc)[:90]}")
                time.sleep(5 * (attempt + 1))
        else:
            print("    the platform itself is refusing everything - wait and")
            print("    rerun; nothing here is wrong with the script.")
            return

        print("\n  probe 2: net position, retried")
        for z in ("SI", "HU"):
            for attempt in range(4):
                try:
                    d = client.query_net_position(z, start=s, end=e,
                                                  dayahead=True)
                    print(f"    query_net_position({z}) OK - {len(d)} rows, "
                          f"mean {float(d.mean()):.0f} MW")
                    print(d.head(3).to_string())
                    return
                except AttributeError:
                    print("    this entsoe-py has no query_net_position")
                    return
                except Exception as exc:                      # noqa: BLE001
                    print(f"    {z} attempt {attempt+1}: "
                          f"{redact(exc)[:90]}")
                    time.sleep(5 * (attempt + 1))
        print("\n    Load works but net position does not, on two zones and")
        print("    four attempts each.  That is the query type, not the")
        print("    platform: script 31 will build net positions by summing")
        print("    scheduled exchanges across each zone's borders instead.")
        return

    print("=" * 76)
    print("NET POSITIONS - all twelve Core bidding zones")
    print("=" * 76)
    for z in OMITTED_CORE:
        t0 = time.time()
        try:
            st = pull(client, "net_position", z, args.start, args.end,
                      RAW, force=args.force)
            print(f"  {z}: {st['written']} written, {st['skipped']} cached, "
                  f"{st['empty']} empty, {st['failed']} failed, "
                  f"{st['rows']:,} rows  ({time.time()-t0:.0f}s)")
        except AttributeError:
            print(f"  {z}: query_net_position not available in this "
                  f"entsoe-py - skipping, script 31 will fall back")
            break
        except Exception as exc:                              # noqa: BLE001
            print(f"  {z}: FAILED {redact(exc)}")

    print("\n" + "=" * 76)
    print("ALEGRO - scheduled exchange on the one Belgium-Germany link")
    print("=" * 76)
    for b in ALEGRO_BORDERS:
        t0 = time.time()
        try:
            st = pull(client, "scheduled", b, args.start, args.end,
                      RAW, force=args.force)
            print(f"  {b}: {st['written']} written, {st['skipped']} cached, "
                  f"{st['rows']:,} rows  ({time.time()-t0:.0f}s)")
        except Exception as exc:                              # noqa: BLE001
            print(f"  {b}: FAILED {redact(exc)}")

    print("\n  Next: scripts/31_close_domain.py joins these onto")
    print("  fb_domain.parquet, turns every fixed_* PTDF into MW, and checks")
    print("  the result against the constraints JAO says actually bound.")


if __name__ == "__main__":
    main()
