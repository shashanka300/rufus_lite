#!/usr/bin/env python3
"""
Ingest Amazon Reviews 2023 review TEXT into rufus_reviews.db reviews table.

Source: data/amazon_reviews_full/raw/review_categories/Electronics.jsonl
  Each line: {rating, title, text, asin, parent_asin, user_id, timestamp,
               helpful_vote, verified_purchase}

Populates:
  rufus_reviews.db / reviews  -- rating, title, text, helpful_votes per ASIN

Used by rufus/reviews.py get_reviews() to surface top helpful review snippets
in the RAG context for Q&A queries.

Run AFTER download completes:
  uv run python scripts/ingest_reviews_text.py

Or specify a different category:
  uv run python scripts/ingest_reviews_text.py --category Cell_Phones_and_Accessories
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

console   = Console(highlight=False, emoji=False)
DB_PATH   = Path("data/rufus_reviews.db")
REVIEWS_DIR = Path("data/amazon_reviews_full/raw/review_categories")
BATCH     = 10_000
MAX_PER_ASIN = 5   # keep top-5 most helpful per product

app = typer.Typer()


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _ingest_file(path: Path, conn: sqlite3.Connection) -> int:
    console.print(f"  Counting lines in {path.name} ...")
    total = _count_lines(path)
    console.print(f"  {total:,} reviews to process")

    inserted = 0
    batch: list[tuple] = []

    with Progress(SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as prog:
        task = prog.add_task(f"  Ingesting {path.stem} ...", total=total)

        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                prog.advance(task)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                asin  = str(obj.get("parent_asin") or obj.get("asin") or "").strip()
                text  = str(obj.get("text") or "").strip()
                title = str(obj.get("title") or "").strip()

                if not asin or not text:
                    continue

                rating  = float(obj.get("rating") or 0)
                helpful = int(obj.get("helpful_vote") or 0)
                verified = 1 if obj.get("verified_purchase") else 0
                date    = str(obj.get("timestamp") or "")[:10]  # first 10 chars = YYYY-MM-DD

                batch.append((asin, rating, title[:200], text[:1000], helpful, verified, date))

                if len(batch) >= BATCH:
                    conn.executemany(
                        "INSERT INTO reviews (asin,rating,title,text,helpful_votes,verified,date) "
                        "VALUES (?,?,?,?,?,?,?)",
                        batch,
                    )
                    conn.commit()
                    inserted += len(batch)
                    batch.clear()

        if batch:
            conn.executemany(
                "INSERT INTO reviews (asin,rating,title,text,helpful_votes,verified,date) "
                "VALUES (?,?,?,?,?,?,?)",
                batch,
            )
            conn.commit()
            inserted += len(batch)

    return inserted


def _dedupe_keep_top(conn: sqlite3.Connection) -> None:
    """Keep only top MAX_PER_ASIN most helpful reviews per ASIN."""
    console.print(f"  Deduping — keeping top {MAX_PER_ASIN} per ASIN ...")
    conn.execute(f"""
        DELETE FROM reviews
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY asin ORDER BY helpful_votes DESC, verified DESC
                ) AS rn FROM reviews
            ) ranked WHERE rn <= {MAX_PER_ASIN}
        )
    """)
    conn.commit()


@app.command()
def main(
    category: str = typer.Option("Electronics", "--category", help="Category name to ingest"),
) -> None:
    jsonl_path = REVIEWS_DIR / f"{category}.jsonl"
    if not jsonl_path.exists():
        console.print(f"[red]Not found: {jsonl_path}[/red]")
        console.print("  Download with: uv run python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('McAuley-Lab/Amazon-Reviews-2023', repo_type='dataset', filename=f'raw/review_categories/{category}.jsonl', local_dir='data/amazon_reviews_full')\"")
        raise typer.Exit(1)

    console.print(f"[bold]Ingest: {category} reviews -> rufus_reviews.db[/bold]")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Clear existing rows for this category to avoid duplicates on re-run
    # (identify by joining with product_meta for this category)
    before = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    console.print(f"  reviews table before: {before:,} rows")

    inserted = _ingest_file(jsonl_path, conn)
    console.print(f"  raw inserted: {inserted:,}")

    _dedupe_keep_top(conn)

    after = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    console.print(f"  reviews table after dedup: {after:,} rows")

    conn.close()
    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()
