"""Fetch a full year of JAO Core flow-based data. Long-running, resumable.

    python scripts/22_fetch_jao_year.py

Everything lands under data/raw/jao/ , which .gitignore already excludes.
Safe to stop with Ctrl-C and restart: every day and every chunk is written to
its own file and skipped on the next run.  Nothing is ever re-downloaded.

TWO PHASES, cheapest and most certain first
-------------------------------------------
Phase 1  extends the JAO_test.py audit from one month to all of 2024, on the
         plain JSON endpoints that are already known to work.  ~365 requests,
         roughly 15 minutes.  This turns the July result into a year, and
         gives an hourly series for the Polish allocation-constraint shadow
         price as a by-product.

Phase 2  starts on `finalComputation` with Presolved=true - the full
         flow-based DOMAIN, not just the constraints that happened to bind.
         That is the input a CNEC implementation actually needs.  It uses a
         different, two-step endpoint: ask for a download, get JSON back with
         a URL in it, then fetch a ZIP from that URL.  Max 2 days per request,
         so ~183 requests for the year.  Volume is unknown until it runs, so
         the script measures the first few chunks and prints a projection.

Phase 1 finishing is the win.  Phase 2 is a bonus that picks up where it
stopped next time.
"""
from pathlib import Path
import io
import sys
import time
import zipfile

import pandas as pd
import requests

BASE = "https://publicationtool.jao.eu/core/api/data/"
FMT  = "%Y-%m-%dT%H:%M:%S.000Z"
FILTER = '{"Presolved":true}'
ROOT = Path("data/raw/jao")

YEAR_START, YEAR_END = "2024-01-01", "2025-01-01"

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

# File downloads need a plain Accept header.  Asking the file route for
# application/json gets an HTML page back instead of the ZIP, which is
# exactly what broke the first attempt at phase 2.
FILES = requests.Session()
FILES.headers.update({
    "Accept": "*/*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})


# ------------------------------------------------------------------
# adaptive throttle: JAO rate-limits, and losing the connection for a
# stretch of days is how the first attempt at this failed.  Back off hard
# on a 429, earn the speed back slowly.
# ------------------------------------------------------------------
class Throttle:
    def __init__(self, pause=1.0, ceiling=20.0):
        self.pause, self.ceiling, self.good = pause, ceiling, 0

    def ok(self):
        self.good += 1
        if self.good >= 25 and self.pause > 1.0:
            self.pause = max(1.0, self.pause / 1.5)
            self.good = 0

    def hit(self, retry_after=None):
        self.good = 0
        self.pause = min(self.ceiling,
                         float(retry_after) if retry_after else self.pause * 2)
        print(f"      rate limited - pausing {self.pause:.0f}s between calls")

    def wait(self):
        time.sleep(self.pause)


T = Throttle()


def get(url, **kw):
    """One request, retrying through rate limits rather than giving up."""
    for attempt in range(6):
        try:
            r = SESSION.get(url, timeout=180, **kw)
            if r.status_code == 429:
                T.hit(r.headers.get("Retry-After"))
                time.sleep(T.pause)
                continue
            if 500 <= r.status_code < 600:
                # JAO throws transient 500s while it builds a file
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            T.ok()
            return r
        except requests.HTTPError:
            raise
        except Exception:                                     # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gave up after 6 attempts: {url}")


# ==================================================================
# PHASE 1 - the JSON endpoints, one day per file
# ==================================================================
# netPos and scheduledExchanges were added after the ENTSO-E platform went
# down mid-migration.  They remove that dependency entirely, and are a better
# source than it was:
#
#   netPos             the hourly CORE net position of every Core zone, plus
#                      the two ALEGrO virtual hubs.  This is the exact
#                      quantity the flow-based constraints act on, published
#                      by the same body that publishes the constraints - so
#                      there is no risk of a definitional mismatch, which
#                      there would have been with an ENTSO-E series.
#   scheduledExchanges every bilateral flow including external borders, as an
#                      independent cross-check on the above.
JSON_ENDPOINTS = ["activeFbConstraints", "priceSpread", "alphaFactor",
                  "allocationConstraint", "maxNetPos", "lta",
                  "netPos", "scheduledExchanges"]


def phase1():
    year = pd.Timestamp(YEAR_START).year
    print("=" * 78)
    print(f"PHASE 1  full-year audit inputs, {year}")
    print("=" * 78)
    days = pd.date_range(YEAR_START, YEAR_END, freq="D",
                         tz="UTC", inclusive="left")

    for ep in JSON_ENDPOINTS:
        folder = ROOT / f"year{pd.Timestamp(YEAR_START).year}" / ep
        folder.mkdir(parents=True, exist_ok=True)
        done = fetched = 0
        t0 = time.time()
        for day in days:
            f = folder / f"{day:%Y%m%d}.csv"
            if f.exists():
                done += 1
                continue
            try:
                r = get(BASE + ep, params={
                    "FromUtc": day.strftime(FMT),
                    "ToUtc":   (day + pd.Timedelta(days=1)).strftime(FMT)})
                payload = r.json()
                rows = payload.get("data", []) if isinstance(payload, dict) \
                    else payload
                pd.DataFrame(rows).to_csv(f, index=False)
                fetched += 1
            except Exception as exc:                          # noqa: BLE001
                print(f"    {ep} {day:%Y-%m-%d}: {str(exc)[:70]}")
            T.wait()
        # stitch the days into one file
        parts = sorted(folder.glob("*.csv"))
        frames = []
        for p in parts:
            try:
                d = pd.read_csv(p)
                if not d.empty:
                    frames.append(d)
            except Exception:                                 # noqa: BLE001
                pass
        if frames:
            big = pd.concat(frames, ignore_index=True)
            out = ROOT / f"{ep}_{pd.Timestamp(YEAR_START).year}.csv"
            big.to_csv(out, index=False)
            print(f"  {ep:22} {len(parts):>3} days "
                  f"({fetched} new, {done} cached)  "
                  f"{len(big):>8,} rows  {time.time()-t0:>5.0f}s  -> {out.name}")
        else:
            print(f"  {ep:22} nothing returned")
    print("\n  PHASE 1 DONE - JAO_test.py can now be pointed at these files.\n")


# ==================================================================
# PHASE 2 - the presolved domain, two days per request, ZIP in two steps
# ==================================================================
def phase2(max_minutes=120):
    """The presolved flow-based domain, fetched by GAP rather than by grid.

    An earlier version walked a fixed two-day grid from 1 January and counted
    files to decide when it was finished.  Running several date windows in
    parallel broke that: each window starts on its own two-day boundary, so
    the files do not line up with one grid, the count overshoots, and it
    reported "185 of 183, complete" with 73 days of June and September
    actually missing.

    So this works out which DAYS are already covered, from the filenames, and
    asks only for what is missing.  That is self-healing: any combination of
    windows, interruptions and retries converges on full coverage, and the
    completion test is coverage rather than a file count.
    """
    year = pd.Timestamp(YEAR_START).year
    folder = ROOT / f"domain{year}"
    folder.mkdir(parents=True, exist_ok=True)

    # A .claim file marks a window some process is fetching right now, and a
    # .failed one marks a window that errored.  Both count as "not available
    # to take", which is what lets several copies of this script run against
    # the SAME year without both starting on the same first missing day.
    # A run killed mid-window leaves a stale .claim: delete *.claim before
    # restarting if nothing is actually running.
    def _days_from(pattern):
        got = set()
        for f in folder.glob(pattern):
            stem = f.name.split(".")[0].replace("fc_", "")
            try:
                d0 = pd.Timestamp(stem)
            except Exception:                                 # noqa: BLE001
                continue
            got.add(d0.date())
            got.add((d0 + pd.Timedelta(days=1)).date())
        return got

    def covered_days():
        """Days genuinely on disk - the real coverage measure."""
        return _days_from("fc_*.csv.gz")

    def taken_days():
        """Coverage plus what other workers have claimed or given up on."""
        return (covered_days() | _days_from("fc_*.claim")
                | _days_from("fc_*.failed"))

    wanted = {d.date() for d in pd.date_range(YEAR_START, YEAR_END, freq="D",
                                              inclusive="left")}
    print("=" * 78)
    print(f"PHASE 2  presolved flow-based domain -> {folder.name}")
    print("=" * 78)
    have = covered_days() & wanted
    print(f"  {len(have)} of {len(wanted)} days already covered; "
          f"{len(wanted) - len(have)} to fetch")

    deadline = time.time() + max_minutes * 60
    sizes, fetched = [], 0
    t0_phase = time.time()

    while True:
        missing = sorted(wanted - taken_days())
        if not missing:
            break
        if time.time() > deadline:
            print("\n  time budget reached - stopping cleanly.")
            break
        s0 = pd.Timestamp(missing[0], tz="UTC")
        e0 = min(s0 + pd.Timedelta(days=2),
                 pd.Timestamp(YEAR_END, tz="UTC"))
        out   = folder / f"fc_{s0:%Y%m%d}.csv.gz"
        claim = folder / f"fc_{s0:%Y%m%d}.claim"
        try:
            # exclusive create: if another worker got here first, move on
            claim.touch(exist_ok=False)
        except FileExistsError:
            continue
        t_chunk = time.time()
        try:
            meta = get(BASE + "finalComputation/download", params={
                "FromUtc":  s0.strftime(FMT),
                "ToUtc":    e0.strftime(FMT),
                "FileType": "csv",
                "Filter":   FILTER}).json()
            if meta.get("rejected"):
                print(f"    {s0:%Y-%m-%d}: rejected - {meta.get('messages')}")
                claim.rename(folder / f'fc_{s0:%Y%m%d}.failed')
                break
            url = meta.get("downloadUrl") or meta.get("DownloadUrl")
            blob = FILES.get(url, timeout=300).content
            if blob[:2] != b"PK":
                print(f"    {s0:%Y-%m-%d}: not a zip, got {blob[:60]!r}")
                claim.rename(folder / f'fc_{s0:%Y%m%d}.failed')
                break
            z = zipfile.ZipFile(io.BytesIO(blob))
            parts = []
            for name in z.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                raw = z.read(name)
                first = raw.split(b"\n", 1)[0].decode("utf-8", "replace")
                sep = ";" if first.count(";") > first.count(",") else ","
                parts.append(pd.read_csv(io.BytesIO(raw), sep=sep,
                                         encoding="utf-8-sig",
                                         low_memory=False))
            if not parts:
                print(f"    {s0:%Y-%m-%d}: zip held no csv")
                claim.rename(folder / f'fc_{s0:%Y%m%d}.failed')
                break
            df = pd.concat(parts, ignore_index=True)
            df.to_csv(out, index=False, compression="gzip")
            claim.unlink(missing_ok=True)
            sizes.append(out.stat().st_size)
            fetched += 1
            left = len(wanted - covered_days())
            print(f"    {s0:%Y-%m-%d}  {len(df):>6,} rows  "
                  f"{out.stat().st_size/1e6:>4.1f} MB  "
                  f"{time.time()-t_chunk:>5.0f}s   "
                  f"{fetched:>3} done, {left:>3} days left, "
                  f"eta {left/2*(time.time()-t0_phase)/max(fetched,1)/60:>4.0f} min")
        except Exception as exc:                              # noqa: BLE001
            print(f"    {s0:%Y-%m-%d}: {type(exc).__name__}: {str(exc)[:70]}")
            # turn the claim into a .failed marker so no worker retries it in
            # this run; delete *.failed and rerun to try those windows again
            claim.rename(folder / f"fc_{s0:%Y%m%d}.failed")
        T.wait()

    stale = list(folder.glob("fc_*.claim"))
    failed = list(folder.glob("fc_*.failed"))
    left = sorted(wanted - covered_days())
    total = sum(f.stat().st_size for f in folder.glob("*.csv.gz"))
    print(f"\n  COVERAGE: {len(wanted)-len(left)} of {len(wanted)} days "
          f"({100*(len(wanted)-len(left))/len(wanted):.1f} %), "
          f"{total/1e6:.0f} MB in {folder.name}")
    if left:
        print(f"  {len(left)} days still missing, first few: "
              f"{[str(d) for d in left[:6]]}")
        print("  rerun to continue - only the gaps are requested")
    if failed:
        print(f"  {len(failed)} window(s) errored; delete *.failed to retry")
    if stale:
        print(f"  {len(stale)} claim(s) still open - normal while other")
        print("  workers are running; delete *.claim if nothing is")
    else:
        print("  complete.")


if __name__ == "__main__":
    # optional: python scripts/22_fetch_jao_year.py 2024-01-01 2024-04-01
    _dates = [a for a in sys.argv[1:] if a[:2] == "20"]
    if len(_dates) == 2:
        YEAR_START, YEAR_END = _dates
        print(f"window restricted to {YEAR_START} .. {YEAR_END}\n")
    ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "--phase2-only" not in sys.argv:
        phase1()
    if "--phase1-only" not in sys.argv:
        phase2()
    print(f"total {(time.time()-t0)/60:.1f} minutes")
