"""Chunked, restartable ENTSO-E puller.

Design notes
------------
* Every (dataset, key, month) is cached as its own parquet under data/raw/.
  Re-running skips anything already on disk, so a pull that dies halfway
  through - rate limit, dropped connection, closed laptop - just needs
  running again.
* Everything is stored with a UTC index. Local market time (CET/CEST) is a
  presentation concern, applied once in the processing step. Mixing the two
  is the classic way to lose an hour in March and duplicate one in October.
* A chunk that genuinely has no data gets an empty marker file, so it is not
  re-requested on every run.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TZ = "UTC"


# --------------------------------------------------------------------------
# fetchers - one per dataset, all sharing the (client, key, start, end) shape
# --------------------------------------------------------------------------

def _load(client, zone, s, e):
    return client.query_load(zone, start=s, end=e)


def _prices(client, zone, s, e):
    return client.query_day_ahead_prices(zone, start=s, end=e)


def _generation(client, zone, s, e):
    return client.query_generation(zone, start=s, end=e)


def _outages(client, zone, s, e):
    return client.query_unavailability_of_generation_units(zone, start=s, end=e)


def _flows(client, border, s, e):
    a, b = border.split(">")
    return client.query_crossborder_flows(a, b, start=s, end=e)


def _scheduled(client, border, s, e):
    """Day-ahead scheduled commercial exchange, MW, one direction.

    This is NOT the same series as physical flow, and the difference is the
    point. Physical flow includes loop flow - power traded between two German
    zones that physically routes through Poland or Czechia. It occupies the
    wire but was never sold across that border, so the day-ahead auction never
    had it to allocate. Sizing market coupling from physical flow hands the
    optimiser capacity the market did not have, which over-couples exactly the
    borders where loop flow is largest.
    """
    a, b = border.split(">")
    return client.query_scheduled_exchanges(a, b, start=s, end=e, dayahead=True)


def _ntc(client, border, s, e):
    a, b = border.split(">")
    return client.query_net_transfer_capacity_dayahead(a, b, start=s, end=e)


def _capacity(client, zone, s, e):
    return client.query_installed_generation_capacity(zone, start=s, end=e)


def _capacity_per_unit(client, zone, s, e):
    return client.query_installed_generation_capacity_per_unit(zone, start=s, end=e)


def _net_position(client, zone, s, e):
    """Day-ahead net position of a whole bidding zone, MW, positive = export.

    Needed for the Core zones this model does not represent. Their PTDF terms
    appear in every flow-based constraint, so their net position has to be
    fixed to what actually happened rather than dropped - see
    scripts/29_build_domain.py.

    Older entsoe-py releases have no query_net_position; the caller falls back
    to summing scheduled exchanges across that zone's borders.
    """
    return client.query_net_position(zone, start=s, end=e, dayahead=True)


ZONE_DATASETS = {
    "load": _load,
    "prices": _prices,
    "generation": _generation,
    "outages": _outages,
    "net_position": _net_position,
}
BORDER_DATASETS = {
    "flows": _flows,
    "scheduled": _scheduled,
    "ntc": _ntc,
}
ANNUAL_DATASETS = {
    "capacity": _capacity,
    "capacity_per_unit": _capacity_per_unit,
}
ALL_DATASETS = {**ZONE_DATASETS, **BORDER_DATASETS, **ANNUAL_DATASETS}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def month_windows(start: str, end: str):
    """[start, end) split into calendar-month windows, UTC."""
    edges = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ), freq="MS")
    return list(zip(edges[:-1], edges[1:]))


def year_windows(start: str, end: str):
    """Calendar-year windows covering [start, end), including a partial final
    year. Without the trailing edge, a period ending mid-2026 would silently
    stop at the 2025 window and you would never fetch the current year - which
    is exactly the register you most want for a capacity snapshot.
    """
    s, e = pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ)
    edges = list(pd.date_range(s, e, freq="YS"))
    if not edges or edges[0] > s:
        edges.insert(0, s)
    if edges[-1] < e:
        edges.append(e)
    return list(zip(edges[:-1], edges[1:]))


def _safe_name(key: str) -> str:
    return key.replace(">", "__to__")


def _normalise(obj) -> pd.DataFrame:
    """Series or DataFrame in, tidy UTC-indexed DataFrame out."""
    df = obj.to_frame() if isinstance(obj, pd.Series) else obj.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" | ".join(str(x) for x in tup if x) for tup in df.columns]
    df.columns = [str(c) for c in df.columns]

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_convert(TZ) if df.index.tz else df.index.tz_localize(TZ)
        df.index.name = "timestamp"
    else:
        # event tables (outages, capacity registers) are not time-indexed
        df = df.reset_index(drop=True)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = df[col].dt.tz_convert(TZ)
                except (TypeError, AttributeError):
                    pass
    return df


_TOKEN_RE = re.compile(r"(securityToken=)[0-9a-fA-F\-]+")


def redact(msg) -> str:
    """Strip the API token out of anything we log. ENTSO-E puts it in the URL,
    so raw error messages carry the credential in plain text."""
    return _TOKEN_RE.sub(r"\1<redacted>", str(msg))


def _is_permanent(exc) -> bool:
    """A 4xx (other than 429 rate limit) will not succeed on retry."""
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code is not None and 400 <= code < 500 and code != 429


def _with_retry(fn, *args, tries: int = 3, base_sleep: int = 3, **kwargs):
    """Retry transient failures only.

    Not retried: a genuine 'no data' answer, or any 4xx short of a rate limit.
    ENTSO-E sometimes answers 400 where it means 'nothing here', and retrying
    those four times just burns two minutes per chunk.
    """
    from entsoe.exceptions import NoMatchingDataError

    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except NoMatchingDataError:
            raise
        except Exception as exc:
            if attempt == tries - 1 or _is_permanent(exc):
                raise
            nap = base_sleep * (2 ** attempt)
            log.warning("    retry %d/%d in %ds (%s)",
                        attempt + 1, tries - 1, nap, redact(exc))
            time.sleep(nap)


# --------------------------------------------------------------------------
# the pull
# --------------------------------------------------------------------------

def pull(client, dataset: str, key: str, start: str, end: str,
         raw_dir: Path, annual: bool = False, force: bool = False) -> dict:
    """Pull one dataset for one zone or border. Returns a summary dict."""
    from entsoe.exceptions import NoMatchingDataError

    fetch = ALL_DATASETS[dataset]
    windows = year_windows(start, end) if annual else month_windows(start, end)
    out_dir = raw_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"written": 0, "skipped": 0, "empty": 0, "failed": 0, "rows": 0}

    for win_start, win_end in windows:
        tag = win_start.strftime("%Y" if annual else "%Y%m")
        path = out_dir / f"{_safe_name(key)}_{tag}.parquet"

        if path.exists() and not force:
            stats["skipped"] += 1
            continue

        t0 = time.time()
        try:
            raw = _with_retry(fetch, client, key, win_start, win_end)
        except NoMatchingDataError:
            pd.DataFrame().to_parquet(path)      # marker: asked, nothing there
            stats["empty"] += 1
            log.info("      %s  no data  (%.0fs)", tag, time.time() - t0)
            continue
        except Exception as exc:
            log.error("      %s  FAILED after %.0fs: %s",
                      tag, time.time() - t0, redact(exc))
            stats["failed"] += 1
            continue

        df = _normalise(raw)
        df.to_parquet(path)
        stats["written"] += 1
        stats["rows"] += len(df)
        log.info("      %s  %d rows  (%.0fs)", tag, len(df), time.time() - t0)

    return stats


def load_raw(dataset: str, raw_dir: Path) -> dict:
    """Reassemble cached chunks into one frame per zone or border."""
    files = sorted((raw_dir / dataset).glob("*.parquet"))
    buckets: dict = {}
    for path in files:
        key = path.stem.rsplit("_", 1)[0].replace("__to__", ">")
        df = pd.read_parquet(path)
        if df.empty:
            continue
        buckets.setdefault(key, []).append(df)
    return {key: pd.concat(frames).sort_index() for key, frames in buckets.items()}
