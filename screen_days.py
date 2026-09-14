"""Screen the twelve candidate sample days for DATA COMPLETENESS only.

    python screen_days.py --year 2025
    python screen_days.py --year 2025 --weekday 2 --nth 2

The sampling rule is the second Wednesday of each month, fixed in advance. This
script does not look at any result. It checks only whether the inputs a run
needs are present for those dates, so that a day can be rejected for a hole in
the data and never for an inconvenient answer. A rejected day is replaced by the
same weekday one week later, and the substitution is printed so it can be
recorded.

WHAT IS AND IS NOT A HOLE
-------------------------
Only quantities that must exist in every hour are tested. Generation by
technology is not: ENTSO-E returns NaN both for a technology that did not run
and for one that was not reported, the two cannot be told apart from the series,
and build() calls fillna(0.0) on them for that reason. French hard coal is
absent from the data all summer because it is idle, and Polish offshore wind is
absent all year because it does not exist; neither is a reason to reject a day.

Switzerland is not a Core zone, so JAO publishes no maxNetPos column for it.
Only the seven Core zones are required.

    hourly frames   24 snapshots present
    load            every zone, every hour, present and positive
    zonal prices    every zone, every hour, present
    fuel prices     the day falls inside the daily series
    flow-based      CNEC rows for all 24 hours of the day
    maxNetPos       24 rows, with a column for each of the seven Core zones

Exit status is 0 if every day passes and 1 if any day is flagged, so it can be
put in front of a batch.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
JAO = ROOT / "data" / "raw" / "jao"

ZONES = ["DE_LU", "FR", "PL", "NL", "BE", "AT", "CH", "CZ"]
CORE_ZONES = ["DE_LU", "FR", "PL", "NL", "BE", "AT", "CZ"]   # CH is not Core

# Frames whose index must cover the day. Contents are checked only where a
# value must exist in every hour: load and zonal prices.
HOURLY = [
    "load.parquet",
    "prices.parquet",
    "generation.parquet",
    "capacity_factors.parquet",
    "net_flows.parquet",
    "boundary_position.parquet",
]

ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}


def nth_weekday(year: int, month: int, weekday: int, nth: int) -> dt.date:
    """The nth occurrence of a weekday in a month. Monday is 0."""
    days = [dt.date(year, month, d)
            for d in range(1, calendar.monthrange(year, month)[1] + 1)
            if dt.date(year, month, d).weekday() == weekday]
    return days[nth - 1]


def load_hourly(name: str):
    path = PROCESSED / name
    if not path.exists():
        return None, f"{name} not found"
    frame = pd.read_parquet(path)
    idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
    return frame.set_axis(idx), None


def _window(frame, day: dt.date):
    start = pd.Timestamp(day, tz="UTC")
    return frame[(frame.index >= start)
                 & (frame.index < start + pd.Timedelta(days=1))]


def _zone_columns(frame, zone: str) -> list:
    """Columns belonging to a zone, whether named 'DE_LU' or 'DE_LU|Tech'."""
    return [c for c in frame.columns
            if str(c) == zone or str(c).startswith(f"{zone}|")]


def check_index(frames: dict, day: dt.date) -> list[str]:
    problems = []
    for name, frame in frames.items():
        if frame is None:
            continue
        n = len(_window(frame, day))
        if n != 24:
            problems.append(f"{name}: {n} of 24 hours")
    return problems


def check_always_present(frames: dict, day: dt.date) -> list[str]:
    """Load and zonal prices exist in every hour of every zone, or the day is
    unusable. Load must also be positive."""
    problems = []
    for name, positive in (("load.parquet", True), ("prices.parquet", False)):
        frame = frames.get(name)
        if frame is None:
            continue
        window = _window(frame, day)
        if len(window) != 24:
            continue                      # already reported by check_index
        for zone in ZONES:
            cols = _zone_columns(window, zone)
            if not cols:
                problems.append(f"{name}: no column for {zone}")
                continue
            col = window[cols[0]]
            if col.isna().any():
                problems.append(f"{name}: {zone} missing "
                                f"{int(col.isna().sum())} hours")
            elif positive and (col <= 0).any():
                problems.append(f"{name}: {zone} has a non-positive value")
    return problems


def check_fuel_prices(day: dt.date) -> list[str]:
    path = PROCESSED / "fuel_prices.parquet"
    if not path.exists():
        return ["fuel_prices.parquet not found"]
    frame = pd.read_parquet(path)
    idx = pd.Series(pd.to_datetime(frame.index, utc=True,
                                   errors="coerce")).dt.date
    if day not in set(idx):
        return ["fuel_prices: date absent from the daily series"]
    row = frame[(idx == day).values]
    empty = [c for c in row.columns if row[c].isna().all()]
    return [f"fuel_prices: NaN {', '.join(map(str, empty))}"] if empty else []


def check_domain(day: dt.date, year: int) -> list[str]:
    path = PROCESSED / f"fb_domain_{year}.parquet"
    if not path.exists():
        return [f"fb_domain_{year}.parquet not found (flow-based runs only)"]
    frame = pd.read_parquet(path, columns=["t"])
    t = pd.to_datetime(frame["t"], utc=True, errors="coerce")
    start = pd.Timestamp(day, tz="UTC")
    window = t[(t >= start) & (t < start + pd.Timedelta(days=1))]
    hours = window.dt.floor("h").nunique()
    if hours != 24:
        return [f"fb_domain: CNECs for {hours} of 24 hours"]
    smallest = int(window.dt.floor("h").value_counts().min())
    if smallest < 10:
        return [f"fb_domain: one hour carries only {smallest} CNEC rows"]
    return []


def check_maxnetpos(day: dt.date, year: int) -> list[str]:
    path = JAO / f"maxNetPos_{year}.csv"
    if not path.exists():
        return [f"maxNetPos_{year}.csv not found (maxNetPos runs only)"]
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [c.strip().lstrip("﻿").lower() for c in frame.columns]
    stamp = next((c for c in frame.columns if "datetime" in c), None)
    if stamp is None:
        return ["maxNetPos: no datetime column"]
    t = pd.to_datetime(frame[stamp], utc=True, errors="coerce")
    start = pd.Timestamp(day, tz="UTC")
    window = frame[(t >= start) & (t < start + pd.Timedelta(days=1))]
    if len(window) != 24:
        return [f"maxNetPos: {len(window)} of 24 rows"]
    missing = [z for z in CORE_ZONES
               if not any(z.lower().split("_")[0] in c for c in window.columns)]
    if missing:
        return [f"maxNetPos: no column for {', '.join(missing)}"]
    return []


def screen(day: dt.date, frames: dict, year: int) -> list[str]:
    return (check_index(frames, day)
            + check_always_present(frames, day)
            + check_fuel_prices(day)
            + check_domain(day, year)
            + check_maxnetpos(day, year))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--weekday", type=int, default=2,
                    help="0 Monday ... 6 Sunday. Default 2, Wednesday.")
    ap.add_argument("--nth", type=int, default=2,
                    help="which occurrence in the month. Default 2.")
    ap.add_argument("--out", default="sample_days.csv")
    a = ap.parse_args()

    frames = {}
    for name in HOURLY:
        frame, note = load_hourly(name)
        if note:
            print(f"  note: {note}")
        frames[name] = frame

    print(f"\nscreening the {ORDINAL.get(a.nth, a.nth)} "
          f"{calendar.day_name[a.weekday].lower()} of each month, {a.year}\n")

    chosen, flagged = [], []
    for month in range(1, 13):
        day = nth_weekday(a.year, month, a.weekday, a.nth)
        problems = screen(day, frames, a.year)
        if not problems:
            print(f"  {day:%a %d %b}   ok")
            chosen.append(day)
            continue

        flagged.append((day, problems))
        print(f"  {day:%a %d %b}   FLAGGED")
        for p in problems:
            print(f"                  {p}")

        stand_in = day + dt.timedelta(days=7)
        if stand_in.month != day.month:
            stand_in = day - dt.timedelta(days=7)
        sub_problems = screen(stand_in, frames, a.year)
        if sub_problems:
            print(f"    substitute {stand_in:%a %d %b} also fails - choose "
                  f"manually and record why")
            for p in sub_problems:
                print(f"                  {p}")
        else:
            print(f"    substitute {stand_in:%a %d %b} ok - using it")
            chosen.append(stand_in)

    out = ROOT / a.out
    pd.DataFrame({"date": [d.isoformat() for d in sorted(chosen)]}).to_csv(
        out, index=False)
    print(f"\n{len(chosen)} of 12 days usable -> {out}")

    if flagged:
        print("\nFlagged days are rejected for missing data only. No result "
              "was inspected in choosing them.")
    return 1 if len(chosen) < 12 else 0


if __name__ == "__main__":
    sys.exit(main())
