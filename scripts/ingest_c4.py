#!/usr/bin/env python3
"""
Ingest Amazon-C4 product metadata into rufus_reviews.db.

Source: data/amazon_c4/sampled_item_metadata_1M.jsonl
  1M lines: {item_id (ASIN), category, metadata (rich description)}

Adds to rufus_reviews.db:
  c4_metadata table  -- ASIN -> category + rich description
  c4_fts virtual table -- FTS5 index for fast text search

The c4 metadata is much richer than the product_meta descriptions for
many ASINs. Used by seller_qa.py and the Q&A intent for grounding answers
in detailed product knowledge.

Run:  uv run python scripts/ingest_c4.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

console = Console(highlight=False, emoji=False)

JSONL_PATH = Path("data/amazon_c4/sampled_item_metadata_1M.jsonl")
DB_PATH    = Path("data/rufus_reviews.db")
BATCH_SIZE = 10_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS c4_metadata (
    asin     TEXT PRIMARY KEY,
    category TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_c4_asin ON c4_metadata(asin);

CREATE VIRTUAL TABLE IF NOT EXISTS c4_fts
    USING fts5(asin UNINDEXED, category, metadata, content=c4_metadata, content_rowid=rowid);
"""


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    console.print("[bold]Ingest: Amazon C4 -> rufus_reviews.db[/bold]")
    if not JSONL_PATH.exists():
        console.print(f"[red]{JSONL_PATH} not found[/red]")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(_SCHEMA)
    conn.commit()

    existing = conn.execute("SELECT COUNT(*) FROM c4_metadata").fetchone()[0]
    console.print(f"  existing c4_metadata rows: {existing:,}")
    if existing > 900_000:
        console.print("  already ingested, skipping")
        conn.close()
        return

    console.print("Counting lines ...")
    total_lines = _count_lines(JSONL_PATH)
    console.print(f"  {total_lines:,} lines in {JSONL_PATH.name}")

    inserted = 0
    skipped  = 0
    batch: list[tuple[str, str, str]] = []

    with Progress(
        SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("  Parsing...", total=total_lines)

        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                progress.advance(task)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                asin     = str(obj.get("item_id", "")).strip()
                category = str(obj.get("category", "")).strip()[:100]
                metadata = str(obj.get("metadata", "")).strip()[:2000]

                if not asin or not metadata:
                    skipped += 1
                    continue

                batch.append((asin, category, metadata))

                if len(batch) >= BATCH_SIZE:
                    conn.executemany(
                        "INSERT OR IGNORE INTO c4_metadata (asin, category, metadata) "
                        "VALUES (?,?,?)",
                        batch,
                    )
                    conn.commit()
                    inserted += len(batch)
                    batch.clear()

        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO c4_metadata (asin, category, metadata) VALUES (?,?,?)",
                batch,
            )
            conn.commit()
            inserted += len(batch)

    console.print(f"  inserted {inserted:,}  skipped {skipped:,}")

    # ── Rebuild FTS index ─────────────────────────────────────────────────────
    console.print("Rebuilding FTS index ...")
    conn.execute("INSERT INTO c4_fts(c4_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()

    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
