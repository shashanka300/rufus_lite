"""
Image URL lookup for product IDs.

Uses data/rufus_images.db (built by scripts/build_image_lookup.py).
Loads the full table into an in-memory dict on first call (~11 MB for 1M rows).
Falls back gracefully when the DB is absent.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("data/rufus_images.db")

_CACHE: dict[str, str] | None = None


def _load() -> dict[str, str]:
    if not DB_PATH.exists():
        return {}
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT product_id, image_url FROM images").fetchall()
    conn.close()
    return dict(rows)


def _ensure() -> dict[str, str]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load()
        print(f"[images] loaded {len(_CACHE):,} image URLs")
    return _CACHE


def get_image_url(product_id: str) -> str | None:
    return _ensure().get(product_id)


def get_image_urls_batch(product_ids: list[str]) -> dict[str, str]:
    cache = _ensure()
    return {pid: cache[pid] for pid in product_ids if pid in cache}


def available() -> bool:
    return DB_PATH.exists()
