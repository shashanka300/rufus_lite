#!/usr/bin/env python3
"""
Ingest Amazon Reviews 2023 -- all categories -- into rufus_reviews.db.

Run after:  uv run python scripts/download_all_data.py catalog --force
            (downloads raw_meta_* and raw_review_* parquets)

Populates
---------
  rufus_reviews.db / product_meta   -- price, rating, features, description,
                                       categories, bought_together per ASIN
  rufus_reviews.db / reviews        -- rating, review_text, helpful_votes
                                       per ASIN (sampled, top-K by helpfulness)

These tables are queried at inference time by rufus/reviews.py to enrich
the RAG context with live price, star rating, and review snippets.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import track

console = Console(highlight=False, emoji=False)

DATA_DIR  = Path("data/amazon_reviews_full")
DB_PATH   = Path("data/rufus_reviews.db")
MAX_REVIEWS_PER_ASIN = 5  # top-5 most helpful reviews per product

_SCHEMA = """
CREATE TABLE IF NOT EXISTS product_meta (
    asin          TEXT PRIMARY KEY,
    title         TEXT,
    price         REAL,
    avg_rating    REAL,
    rating_count  INTEGER,
    features      TEXT,          -- JSON list of bullet points
    description   TEXT,
    category      TEXT,
    store         TEXT,
    bought_together TEXT,        -- JSON list of ASINs
    updated_at    TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    asin          TEXT,
    rating        REAL,
    title         TEXT,
    text          TEXT,
    helpful_votes INTEGER DEFAULT 0,
    verified      INTEGER DEFAULT 0,
    date          TEXT
);
CREATE INDEX IF NOT EXISTS idx_meta_asin    ON product_meta(asin);
CREATE INDEX IF NOT EXISTS idx_reviews_asin ON reviews(asin);
"""


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    return conn


def _safe_json(val) -> str | None:
    if val is None:
        return None
    import json
    try:
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        return json.dumps(list(val))
    except Exception:
        return str(val)


def _ingest_meta(path: Path, conn: sqlite3.Connection) -> int:
    """Ingest one raw_meta_* parquet file."""
    try:
        df = pd.read_parquet(path, columns=[
            "parent_asin", "title", "price", "average_rating", "rating_number",
            "features", "description", "store", "categories", "bought_together",
        ])
    except Exception as e:
        console.print(f"  [red]skip[/red] {path.name}: {e}")
        return 0

    df = df.rename(columns={"parent_asin": "asin", "average_rating": "avg_rating",
                             "rating_number": "rating_count"})
    df["features"]        = df["features"].apply(_safe_json)
    df["description"]     = df["description"].apply(
        lambda x: str(x[0])[:500] if isinstance(x, list) and len(x) > 0
                  else (str(x)[:500] if x is not None and not (hasattr(x, '__len__') and len(x) == 0) else None)
    )
    df["category"]        = df["categories"].apply(
        lambda x: " > ".join(x[:3]) if isinstance(x, list) and x else None
    )
    df["bought_together"] = df["bought_together"].apply(_safe_json)
    df["price"]           = pd.to_numeric(df["price"], errors="coerce")
    from datetime import datetime
    df["updated_at"] = datetime.utcnow().isoformat()

    rows = df[["asin","title","price","avg_rating","rating_count",
               "features","description","category","store","bought_together","updated_at"]].values.tolist()
    conn.executemany(
        """INSERT OR REPLACE INTO product_meta
           (asin,title,price,avg_rating,rating_count,features,description,
            category,store,bought_together,updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def _ingest_reviews(path: Path, conn: sqlite3.Connection) -> int:
    """Ingest one raw_review_* parquet, keeping top-K most helpful per ASIN."""
    try:
        df = pd.read_parquet(path, columns=[
            "parent_asin", "rating", "title", "text",
            "helpful_vote", "verified_purchase", "timestamp",
        ])
    except Exception as e:
        try:
            df = pd.read_parquet(path, columns=[
                "parent_asin", "rating", "title", "text",
            ])
            df["helpful_vote"] = 0
            df["verified_purchase"] = False
            df["timestamp"] = None
        except Exception as e2:
            console.print(f"  [red]skip review[/red] {path.name}: {e2}")
            return 0

    df = df.rename(columns={"parent_asin": "asin", "helpful_vote": "helpful_votes",
                             "verified_purchase": "verified"})
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype(int)
    df["verified"]      = df["verified"].astype(int)
    df["text"]          = df["text"].fillna("").str.slice(0, 600)
    df["title"]         = df["title"].fillna("").str.slice(0, 120)
    df["date"]          = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")

    # Keep top-K most helpful per ASIN
    df = (df.sort_values("helpful_votes", ascending=False)
            .groupby("asin", group_keys=False)
            .head(MAX_REVIEWS_PER_ASIN))

    rows = df[["asin","rating","title","text","helpful_votes","verified","date"]].values.tolist()
    conn.executemany(
        """INSERT OR IGNORE INTO reviews
           (asin,rating,title,text,helpful_votes,verified,date)
           VALUES (?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def main() -> None:
    console.print("[bold]Ingest: Amazon Reviews 2023 (all categories) -> rufus_reviews.db[/bold]")
    conn = _get_db()

    meta_files   = sorted(DATA_DIR.glob("raw_meta_*/*.parquet"))
    review_files = sorted(DATA_DIR.glob("raw_review_*/*.parquet"))

    console.print(f"  meta files:   {len(meta_files)}")
    console.print(f"  review files: {len(review_files)}")

    if not meta_files:
        console.print("[red]No raw_meta_* files found. Run: uv run python scripts/download_all_data.py catalog --force[/red]")
        return

    total_meta = 0
    for f in track(meta_files, description="Ingesting metadata..."):
        total_meta += _ingest_meta(f, conn)
    console.print(f"  [green]meta rows: {total_meta:,}[/green]")

    total_reviews = 0
    for f in track(review_files, description="Ingesting reviews..."):
        total_reviews += _ingest_reviews(f, conn)
    console.print(f"  [green]review rows: {total_reviews:,}[/green]")

    # Summary
    for tbl in ("product_meta", "reviews"):
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        console.print(f"  {tbl}: {n:,} rows")

    conn.close()
    console.print("[bold green]Amazon Reviews ingest complete.[/bold green]")


if __name__ == "__main__":
    main()
