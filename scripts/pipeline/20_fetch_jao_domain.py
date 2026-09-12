"""Download the published CORE flow-based net-position bounds from JAO.

    .venv-data\\Scripts\\activate
    python scripts/20_fetch_jao_domain.py --years 2024 2025
    python scripts/20_fetch_jao_domain.py --years 2024 --probe

Runs in .venv-data alongside 06_fetch_fuel_prices.py - it needs requests, not
pypsa. So does 21_build_hourly_net_position.py. Deactivate and return to .venv
before running 10_build_network.py, which does need pypsa.

What this is, and why it changes the model
------------------------------------------
CORE is the capacity calculation region covering thirteen central European
bidding zones. Since June 2022 its day-ahead capacity is allocated flow-based:
instead of an NTC per border, the TSOs publish a set of critical network
elements, each with a zonal PTDF vector and a remaining available margin, and
the market clears against

    sum over zones of PTDF(zone, element) * net_position(zone) <= RAM(element)

for every element, recomputed EVERY HOUR. Prices separate when an element
binds, and its shadow price reaches a border through the PTDF difference - so
two zones can clear apart while the interconnector between them sits at half
load. That is why price separation and flow saturation came apart in
validate.price_separation(), and why a static per-border limit cannot
reproduce the timing.

The model's net-position bounds are annual quantiles of observed flow: one
number per zone per direction, constant for the year. The domain they stand in
for moves hourly. JAO publishes that hourly movement, and this pulls it.

    https://publicationtool.jao.eu/core/api/data/maxNetPos

Data begins at business day 2022-06-09. Switzerland is NOT in CORE - it is
outside EU market coupling entirely - so it keeps its static bound.

Everything here is a public series. Nothing from work goes in this repository.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402
import requests                                              # noqa: E402

BASE = "https://publicationtool.jao.eu/core/api/data/"
FMT = "%Y-%m-%dT%H:%M:%S.000Z"
RAW = ROOT / "data" / "raw" / "jao"


class RateLimited(Exception):
    """A 429. Kept separate because it is answered by waiting, not by giving up."""

    def __init__(self, retry_after: float):
        super().__init__(f"429, retry after {retry_after:.0f}s")
        self.retry_after = retry_after


def fetch(endpoint: str, start: pd.Timestamp, end: pd.Timestamp,
          session: requests.Session) -> list:
    """One GET for ONE DAY. Returns the row list out of the API's wrapper.

    The window is one day whether you like it or not: a seven-day range returns
    400 Bad Request, with or without Skip/Take. That is why a year costs 366
    requests and why the rate limit below is the real constraint.

    The response is {"data": [...], "rejected": false, "messages": null}.
    """
    r = session.get(BASE + endpoint, timeout=120, params={
        "FromUtc": start.strftime(FMT), "ToUtc": end.strftime(FMT)})
    if r.status_code == 429:
        hdr = r.headers.get("Retry-After", "")
        try:
            wait = float(hdr)
        except ValueError:
            wait = 60.0
        raise RateLimited(max(wait, 30.0))
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict):
        for key in ("data", "Data", "items", "results"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        return [payload]
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--endpoint", default="maxNetPos")
    ap.add_argument("--probe", action="store_true",
                    help="fetch one day, print the schema, write nothing")
    ap.add_argument("--refetch", action="store_true",
                    help="ignore what is already on disk and start over")
    ap.add_argument("--pause", type=float, default=1.0,
                    help="seconds between requests - a public service. This is "
                         "a FLOOR; the loop slows itself down after a 429.")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop after this long and leave the rest to a resumed "
                         "run. 0 means run to the end.")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    if args.probe:
        day = pd.Timestamp(f"{args.years[0]}-06-15", tz="UTC")
        rows = fetch(args.endpoint, day, day + pd.Timedelta(days=1), session)
        print(f"  {len(rows)} rows for {day.date()}")
        if rows:
            print("\n  keys:")
            for k in sorted(rows[0]):
                print(f"    {k!r:32} = {rows[0][k]!r}")
        return

    # ONE BUSINESS DAY PER REQUEST. A multi-day window is accepted and then
    # silently truncated: a seven-day request returned two days of rows, and a
    # whole year came back as 48 rows dated the 30th of December. Nothing
    # errored - the data was simply not there. Hence the coverage guard below;
    # a fetch that quietly returns 0.5% of a year is worse than one that fails.
    for year in args.years:
        out = RAW / f"{args.endpoint}_{year}.json"
        days = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01",
                             freq="D", tz="UTC", inclusive="left")
        # Resume. A first pass returned 250 of 366 days - scattered failures,
        # not a contiguous gap - so refetching the whole year to recover a
        # third of it wastes the service's time and ours.
        rows: list = []
        have: set = set()
        if out.exists() and not args.refetch:
            try:
                rows = json.loads(out.read_text(encoding="utf-8"))
                have = {str(r.get("dateTimeUtc", ""))[:10] for r in rows
                        if isinstance(r, dict)}
                have.discard("")
                print(f"  {year}: resuming, {len(rows):,} rows for "
                      f"{len(have)} days already on disk")
            except Exception:                                 # noqa: BLE001
                rows, have = [], set()

        todo = [d for d in days if d.strftime("%Y-%m-%d") not in have]
        print(f"  {year}: {len(todo)} days to fetch")
        # ADAPTIVE THROTTLE. The service allows roughly a hundred requests
        # before it starts answering 429, and a fixed short backoff turns that
        # into thirty consecutive lost days - which is exactly what the first
        # pass did. So: wait properly when told to, and slow the whole loop
        # down after a refusal rather than sprinting back into one.
        failed: list = []
        pause = args.pause
        streak = 0
        t_start = time.time()
        for i, day in enumerate(todo):
            if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
                print(f"  stopping at --max-minutes {args.max_minutes}; "
                      f"re-run to resume from here")
                break
            got, err = [], None
            for attempt in (1, 2, 3, 4):
                try:
                    got = fetch(args.endpoint, day, day + pd.Timedelta(days=1),
                                session)
                    err = None
                    break
                except RateLimited as rl:
                    err = str(rl)
                    pause = min(pause * 2, 20.0)             # slow down globally
                    streak = 0
                    if attempt < 4:
                        time.sleep(rl.retry_after * attempt)
                except Exception as exc:                      # noqa: BLE001
                    err = str(exc)[:80]
                    if attempt < 4:
                        time.sleep(pause * 4 * attempt)
            if err:
                failed.append((day.date(), err))
            else:
                streak += 1
                if streak >= 25:                              # earn speed back
                    pause = max(pause * 0.8, args.pause)
                    streak = 0
            rows.extend(got)
            if i % 30 == 0 or i == len(todo) - 1:
                print(f"  {day.date()}  total {len(rows):,} rows, "
                      f"{len(failed)} failed days, pause {pause:.1f}s, "
                      f"{(time.time() - t_start) / 60:.0f} min elapsed")
            time.sleep(pause)

        # Coverage, measured on distinct timestamps rather than row count.
        stamps = {r.get("dateTimeUtc") for r in rows if isinstance(r, dict)}
        stamps.discard(None)
        expected = len(days) * 24
        cover = 100 * len(stamps) / expected if expected else 0.0
        print(f"\n  {year}: {len(rows):,} rows, {len(stamps):,} distinct hours "
              f"of {expected:,} = {cover:.1f}% coverage, "
              f"{len(failed)} failed days")
        for d, msg in failed[:5]:
            print(f"    {d}: {msg}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")

        if cover < 50:
            print(f"  {year}: REFUSING to write - coverage below 50%. The file "
                  f"would look fine and be almost empty.")
            continue
        # Deduplicate: a resumed day that was partially present would otherwise
        # appear twice and be counted twice in the median.
        seen, unique = set(), []
        for r in rows:
            k = r.get("dateTimeUtc") if isinstance(r, dict) else None
            if k in seen:
                continue
            seen.add(k)
            unique.append(r)
        rows = unique
        out.write_text(json.dumps(rows), encoding="utf-8")
        print(f"  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)\n")


if __name__ == "__main__":
    main()
