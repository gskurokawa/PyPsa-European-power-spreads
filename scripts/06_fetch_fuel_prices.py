"""Download the real gas, carbon and coal series into data/raw/fuel/.

    .venv-data\\Scripts\\activate
    python scripts/06_fetch_fuel_prices.py
    python scripts/06_fetch_fuel_prices.py --years 2024 2025 2026

Runs in .venv-data, not .venv - see requirements-data.txt for why.

Sources, and why each one
-------------------------
carbon   EEX EU ETS primary auction reports. The clearing price of the actual
         auction, published by the auction platform itself: a primary source,
         citable, free, and complete back to 2012. Preferred over a scraped
         futures series precisely because a reviewer can check it.
gas      Dutch TTF front-month (Yahoo `TTF=F`), already quoted in EUR/MWh, so
         no conversion and no calorific assumption.
coal     API2 CIF ARA front-month (Yahoo `MTF=F`) in USD/tonne, converted
         downstream at 6.978 MWh_th/tonne. The IMF Primary Commodity Price
         System monthly series is also pulled, as an independent check that the
         daily futures series has not drifted - two sources disagreeing by more
         than a few percent means one of them is wrong.
FX       EURUSD (Yahoo `EURUSD=X`), replacing the constant 1.08 the fallback
         used, which is worth roughly 8% on the coal cost in 2025 alone.

Everything written here is a public series. Nothing from work goes in this
repository.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402
import requests                                              # noqa: E402

from spread.config import RAW, load_config                   # noqa: E402

FUEL = RAW / "fuel"

EEX_URL = ("https://public.eex-group.com/eex/eua-auction-report/"
           "emission-spot-primary-market-auction-report-{year}-data.xlsx")
IMF_URL = ("https://www.imf.org/-/media/files/research/commodityprices/"
           "monthly/external-data.xlsx")
HEADERS = {"User-Agent": "Mozilla/5.0 (research; fuel price download)"}

TICKERS = {
    "ttf.csv":  ("TTF=F",    "price_eur_mwh_th"),
    "coal.csv": ("MTF=F",    "price_usd_t"),
    "fx.csv":   ("EURUSD=X", "usd_per_eur"),
}


# --------------------------------------------------------------------------
# Yahoo
# --------------------------------------------------------------------------
def fetch_yahoo(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: no rows returned")
    if isinstance(df.columns, pd.MultiIndex):       # newer yfinance always does
        df.columns = df.columns.get_level_values(0)
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close[~close.index.duplicated(keep="last")].sort_index()


# --------------------------------------------------------------------------
# EEX auction reports
# --------------------------------------------------------------------------
def _locate_header(raw: pd.DataFrame) -> int | None:
    """EEX puts a title block above the table; find the real header row."""
    for i in range(min(40, len(raw))):
        cells = [str(c).lower() for c in raw.iloc[i].tolist()]
        joined = " | ".join(cells)
        if "auction price" in joined and "date" in joined:
            return i
    return None


def fetch_eex_year(year: int) -> pd.Series:
    url = EEX_URL.format(year=year)
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    book = pd.ExcelFile(io.BytesIO(r.content))

    for sheet in book.sheet_names:
        raw = book.parse(sheet, header=None)
        row = _locate_header(raw)
        if row is None:
            continue
        df = book.parse(sheet, header=row)
        df.columns = [str(c).strip() for c in df.columns]

        date_col = next((c for c in df.columns
                         if "date" in c.lower() and "settlement" not in c.lower()), None)
        price_col = next((c for c in df.columns if "auction price" in c.lower()), None)
        if date_col is None or price_col is None:
            continue

        # EUAA (aviation) auctions clear separately and are a different market.
        name_col = next((c for c in df.columns
                         if "auction name" in c.lower() or c.lower() == "auction"), None)
        if name_col is not None:
            names = df[name_col].astype(str).str.upper()
            keep = names.str.contains("EUA") & ~names.str.contains("EUAA")
            if keep.any():
                df = df[keep]

        s = pd.Series(
            pd.to_numeric(df[price_col], errors="coerce").values,
            index=pd.to_datetime(df[date_col], errors="coerce"),
        ).dropna()
        s = s[~s.index.isna()]
        if not s.empty:
            s.index = s.index.tz_localize(None).normalize()
            return s[~s.index.duplicated(keep="last")].sort_index()

    raise RuntimeError(f"{year}: no usable table in {book.sheet_names}")


# --------------------------------------------------------------------------
# IMF cross-check
# --------------------------------------------------------------------------
def fetch_imf_coal() -> pd.DataFrame:
    """Monthly IMF coal series, USD/tonne. A check on the futures series."""
    r = requests.get(IMF_URL, headers=HEADERS, timeout=120)
    r.raise_for_status()
    book = pd.ExcelFile(io.BytesIO(r.content))
    raw = book.parse(book.sheet_names[0], header=None)

    header = next((i for i in range(min(10, len(raw)))
                   if raw.iloc[i].astype(str).str.contains("Coal", case=False).any()),
                  0)
    df = book.parse(book.sheet_names[0], header=header)
    df.columns = [str(c).strip() for c in df.columns]

    coal_cols = [c for c in df.columns if "coal" in c.lower()]
    if not coal_cols:
        raise RuntimeError(f"no coal column among {list(df.columns)[:12]}")

    # IMF stamps months as e.g. 2024M1 in the first column.
    stamps = df[df.columns[0]].astype(str).str.strip()
    idx = pd.to_datetime(stamps.str.replace("M", "-", regex=False),
                         format="%Y-%m", errors="coerce")
    out = df[coal_cols].apply(pd.to_numeric, errors="coerce")
    out.index = idx
    return out[~out.index.isna()].sort_index()


# --------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    start = str(pd.Timestamp(cfg["period"]["start"]).date())
    end = str(pd.Timestamp(cfg["period"]["end"]).date())
    default_years = list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))

    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=default_years,
                    help="EEX auction report years to download")
    ap.add_argument("--skip-imf", action="store_true")
    args = ap.parse_args()

    FUEL.mkdir(parents=True, exist_ok=True)
    report = []

    # ---- Yahoo series ----
    for filename, (ticker, colname) in TICKERS.items():
        try:
            s = fetch_yahoo(ticker, start, end)
            s.rename(colname).rename_axis("date").to_frame().to_csv(FUEL / filename)
            report.append((filename, f"{ticker}: {len(s)} rows, "
                                     f"{s.index[0].date()} to {s.index[-1].date()}",
                           float(s.mean())))
        except Exception as exc:                              # noqa: BLE001
            report.append((filename, f"FAILED {ticker}: {exc}", float("nan")))

    # ---- EEX auctions ----
    parts, notes = [], []
    for year in args.years:
        try:
            s = fetch_eex_year(year)
            parts.append(s)
            notes.append(f"{year}: {len(s)} auctions")
        except Exception as exc:                              # noqa: BLE001
            notes.append(f"{year}: FAILED ({exc})")
    if parts:
        eua = pd.concat(parts).sort_index()
        eua = eua[~eua.index.duplicated(keep="last")]
        eua.rename("auction_price_eur_t").rename_axis("date").to_frame().to_csv(
            FUEL / "eua.csv")
        report.append(("eua.csv", "EEX " + "; ".join(notes), float(eua.mean())))
    else:
        report.append(("eua.csv", "EEX " + "; ".join(notes), float("nan")))

    # ---- IMF cross-check ----
    imf = None
    if not args.skip_imf:
        try:
            imf = fetch_imf_coal()
            imf.to_csv(FUEL / "imf_coal_monthly.csv")
        except Exception as exc:                              # noqa: BLE001
            print(f"IMF cross-check unavailable: {exc}")

    # ---- report ----
    print("\n" + "=" * 78)
    print("DOWNLOADED")
    print("=" * 78)
    for name, note, mean in report:
        mean_txt = "-" if pd.isna(mean) else f"mean {mean:,.2f}"
        print(f"  {name:<22} {note:<46} {mean_txt}")

    if imf is not None and (FUEL / "coal.csv").exists():
        print("\n" + "=" * 78)
        print("COAL CROSS-CHECK: API2 futures vs IMF monthly, USD/tonne")
        print("=" * 78)
        api2 = pd.read_csv(FUEL / "coal.csv", index_col=0, parse_dates=True)
        api2_m = api2.iloc[:, 0].resample("MS").mean()
        window = imf.reindex(api2_m.index).dropna(how="all")
        if not window.empty:
            comp = pd.DataFrame({"api2": api2_m.reindex(window.index)})
            for c in window.columns:
                comp[c[:28]] = window[c]
            print(comp.round(1).tail(18).to_string())
            print("\n  Different coals - Australian and South African are not API2 -")
            print("  so the levels differ. What matters is that they move together.")

    print("\n  Next: python scripts/07_build_fuel_prices.py   (in .venv)")


if __name__ == "__main__":
    main()
