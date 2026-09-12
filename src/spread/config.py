"""Paths, scope config and API credentials."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_config(path: Path | None = None) -> dict:
    """Read config/zones.yaml."""
    path = path or ROOT / "config" / "zones.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def api_key() -> str:
    """ENTSO-E token from .env. Never hardcode it, never commit it.

    dotenv is imported here rather than at module level so that scripts which
    never touch the API - the fuel prices, the fleet - can import this module
    in an environment that does not have it.
    """
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.getenv("ENTSOE_API_KEY")
    if not key or key.startswith("paste_"):
        raise RuntimeError(
            "ENTSOE_API_KEY is not set.\n"
            "Copy .env.example to .env and paste your Transparency Platform token into it."
        )
    return key


def tee_output(name: str):
    """Send stdout to the terminal AND to logs/<name>.txt.

    Every one of these scripts prints hundreds of lines of which the
    interesting twenty are scattered through the middle. Keeping a copy on disk
    means a report can be read after the fact rather than re-run - which cost
    an eight-minute sweep the first time only one script had it.

    Call once at the top of main().
    """
    import sys

    class _Tee:
        def __init__(self, stream, path):
            self.stream = stream
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(path, "w", encoding="utf-8")

        def write(self, data):
            self.stream.write(data)
            self.file.write(data)
            return len(data)

        def flush(self):
            self.stream.flush()
            self.file.flush()

    path = ROOT / "logs" / f"{name}.txt"
    sys.stdout = _Tee(sys.__stdout__, path)
    import logging
    root = logging.getLogger()
    if root.handlers:
        root.handlers[0].stream = sys.stdout
    return path
