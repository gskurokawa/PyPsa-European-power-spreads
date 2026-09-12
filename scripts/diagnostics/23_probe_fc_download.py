"""Find out what the finalComputation download endpoint actually returns.

    python scripts/23_probe_fc_download.py

Prints, and does not interpret.  Nothing is cached and nothing large is
fetched - four probes, a few seconds each.

Why: phase 2 of script 22 failed on every chunk with pandas complaining that
a CSV had 5 fields where it expected 1, and one attempt returned a 500 whose
URL was the bare site root.  Both are the signature of an HTML page being
returned where a file was expected.  This prints the bytes so the fix is
based on evidence rather than another guess.
"""
import json

import requests

BASE = "https://publicationtool.jao.eu/core/api/data/"
FROM = "2024-01-01T00:00:00.000Z"
TO   = "2024-01-03T00:00:00.000Z"
FILT = '{"Presolved":true}'


def show(label, r, body_chars=400):
    print(f"\n--- {label}")
    print(f"    status        {r.status_code}")
    print(f"    final url     {r.url}")
    if r.history:
        print(f"    redirects     {[h.status_code for h in r.history]} "
              f"-> {[h.headers.get('Location') for h in r.history]}")
    print(f"    content-type  {r.headers.get('Content-Type')}")
    print(f"    length        {r.headers.get('Content-Length')} "
          f"(actual {len(r.content)} bytes)")
    print(f"    disposition   {r.headers.get('Content-Disposition')}")
    head = r.content[:body_chars]
    try:
        print(f"    first bytes   {head.decode('utf-8', 'replace')!r}")
    except Exception:                                         # noqa: BLE001
        print(f"    first bytes   {head!r}")


# ==================================================================
print("=" * 78)
print("PROBE 1  ask for the download, look at the JSON we get back")
print("=" * 78)
s = requests.Session()
s.headers.update({"Accept": "application/json"})
r1 = s.get(BASE + "finalComputation/download", timeout=120, params={
    "FromUtc": FROM, "ToUtc": TO, "FileType": "csv", "Filter": FILT})
show("download request", r1)
meta = None
try:
    meta = r1.json()
    print("\n    parsed JSON:")
    print("   ", json.dumps(meta, indent=2)[:1200])
except Exception as exc:                                      # noqa: BLE001
    print(f"    not JSON: {exc}")


# ==================================================================
print("\n" + "=" * 78)
print("PROBE 2  follow the downloadUrl, with a browser-ish Accept header")
print("=" * 78)
if isinstance(meta, dict):
    url = meta.get("downloadUrl") or meta.get("DownloadUrl") or ""
    print(f"    downloadUrl as given: {url!r}")
    if url and not url.lower().startswith("http"):
        url = "https://publicationtool.jao.eu" + \
              ("" if url.startswith("/") else "/") + url
        print(f"    made absolute       : {url}")
    if url:
        s2 = requests.Session()
        s2.headers.update({
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            r2 = s2.get(url, timeout=180, allow_redirects=True)
            show("file fetch", r2)
        except Exception as exc:                              # noqa: BLE001
            print(f"    failed: {exc}")
    else:
        print("    no downloadUrl field present")
else:
    print("    skipped - probe 1 returned no JSON")


# ==================================================================
print("\n" + "=" * 78)
print("PROBE 3  can the ORDINARY json endpoint take a Presolved filter?")
print("   (if yes, the whole ZIP business is unnecessary - phase 2 becomes")
print("    day-by-day json exactly like phase 1)")
print("=" * 78)
for label, params in [
    ("no filter, 1 hour", {
        "FromUtc": "2024-01-01T00:00:00.000Z",
        "ToUtc":   "2024-01-01T01:00:00.000Z"}),
    ("Filter=Presolved, 1 hour", {
        "FromUtc": "2024-01-01T00:00:00.000Z",
        "ToUtc":   "2024-01-01T01:00:00.000Z",
        "Filter":  FILT}),
    ("Presolved=true, 1 hour", {
        "FromUtc":   "2024-01-01T00:00:00.000Z",
        "ToUtc":     "2024-01-01T01:00:00.000Z",
        "Presolved": "true"}),
]:
    try:
        r = s.get(BASE + "finalComputation", timeout=180, params=params)
        n = "?"
        try:
            p = r.json()
            rows = p.get("data", p) if isinstance(p, dict) else p
            n = len(rows)
            cols = list(rows[0]) if n else []
        except Exception:                                     # noqa: BLE001
            cols = []
        print(f"\n    {label:26} status {r.status_code}  rows {n}  "
              f"{len(r.content):,} bytes")
        if cols:
            print(f"       columns: {', '.join(map(str, cols[:14]))} ...")
    except Exception as exc:                                  # noqa: BLE001
        print(f"\n    {label:26} failed: {str(exc)[:90]}")


# ==================================================================
print("\n" + "=" * 78)
print("PROBE 4  how big is one presolved DAY, if probe 3 worked?")
print("=" * 78)
try:
    r = s.get(BASE + "finalComputation", timeout=300, params={
        "FromUtc": "2024-01-01T00:00:00.000Z",
        "ToUtc":   "2024-01-02T00:00:00.000Z",
        "Filter":  FILT})
    p = r.json()
    rows = p.get("data", p) if isinstance(p, dict) else p
    print(f"    status {r.status_code}   {len(rows):,} rows   "
          f"{len(r.content)/1e6:.1f} MB of json")
    print(f"    projected year: {len(rows)*366:,} rows, "
          f"{len(r.content)*366/1e9:.2f} GB of json "
          f"(much smaller as csv/parquet)")
except Exception as exc:                                      # noqa: BLE001
    print(f"    failed: {str(exc)[:120]}")

print("\ndone - paste all of the above")
