#!/usr/bin/env python3
"""
Build data/rufus_images.db — product_id -> image_url lookup table.

Sources combined (SQID takes priority over Amazon Reviews):
  1. SQID product_image_urls.parquet + supp_product_image_urls.parquet  (182 K)
  2. Amazon Reviews 2023 raw_meta_* parquets — large image from images column

Run once:
  uv run python scripts/build_image_lookup.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import track

console = Console(highlight=False, emoji=False)

DB_PATH    = Path("data/rufus_images.db")
SQID_DIR   = Path("data/sqid/data")
AR_DIR     = Path("data/amazon_reviews_full")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    product_id TEXT PRIMARY KEY,
    image_url  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_img_pid ON images(product_id);
"""


def _extract_large_url(img) -> str | None:
    """Pull the first non-null 'large' (then 'hi_res') URL from the images dict."""
    if img is None or (isinstance(img, float)):
        return None
    try:
        if isinstance(img, str):
            img = json.loads(img.replace("'", '"'))
        if not isinstance(img, dict):
            return None
        for key in ("large", "hi_res"):
            urls = img.get(key) or []
            for url in urls:
                if url and isinstance(url, str) and url.startswith("http"):
                    return url
    except Exception:
        pass
    return None


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def _upsert(conn: sqlite3.Connection, rows: list[tuple[str, str]]) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO images (product_id, image_url) VALUES (?, ?)",
        rows,
    )
    conn.commit()


def main() -> None:
    console.print(f"[bold]Building image lookup DB -> {DB_PATH}[/bold]")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    _init_db(conn)

    total_before = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    console.print(f"  existing rows: {total_before:,}")

    # ── 1. SQID image URLs (highest quality, priority) ────────────────────────
    console.print("\n[cyan]1. SQID image URLs[/cyan]")
    sqid_rows: list[tuple[str, str]] = []
    for fname in ("product_image_urls.parquet", "supp_product_image_urls.parquet"):
        p = SQID_DIR / fname
        if not p.exists():
            console.print(f"  [yellow]missing {p}[/yellow]")
            continue
        df = pd.read_parquet(p).dropna(subset=["image_url"])
        df = df[df["image_url"].str.startswith("http", na=False)]
        sqid_rows.extend(zip(df["product_id"].tolist(), df["image_url"].tolist()))
        console.print(f"  {fname}: {len(df):,} rows")

    sqid_rows = list({pid: url for pid, url in sqid_rows}.items())  # dedupe, keep last
    _upsert(conn, sqid_rows)
    console.print(f"  inserted {len(sqid_rows):,} SQID images")

    # ── 2. Amazon Reviews 2023 raw_meta_* parquets ────────────────────────────
    console.print("\n[cyan]2. Amazon Reviews parquets[/cyan]")
    parquets = sorted(AR_DIR.rglob("*.parquet"))
    console.print(f"  found {len(parquets)} parquet files")

    ar_total = 0
    for pq in track(parquets, description="  Parsing..."):
        try:
            df = pd.read_parquet(pq, columns=["parent_asin", "images"])
        except Exception as e:
            console.print(f"  [red]skip {pq.name}: {e}[/red]")
            continue

        rows: list[tuple[str, str]] = []
        for pid, img in zip(df["parent_asin"], df["images"]):
            if not isinstance(pid, str) or not pid:
                continue
            url = _extract_large_url(img)
            if url:
                rows.append((pid, url))

        if rows:
            # INSERT OR IGNORE — SQID already inserted, this won't overwrite
            _upsert(conn, rows)
            ar_total += len(rows)

    console.print(f"  inserted {ar_total:,} Amazon Reviews images")

    total_after = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    conn.close()

    console.print(f"\n[bold green]Done.[/bold green]  "
                  f"Total unique product images: {total_after:,}  "
                  f"(+{total_after - total_before:,} new)")


if __name__ == "__main__":
    main()
