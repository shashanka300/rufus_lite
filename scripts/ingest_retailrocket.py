#!/usr/bin/env python3
"""
Ingest RetailRocket session events into rufus_personalization.db.

Source: data/personalization/retailrocket/events.csv
  ~4.7M rows: timestamp, visitorid, event (view/addtocart/transaction), itemid

Populates:
  item_popularity  -- view/cart/purchase counts + composite score per item
  co_view         -- items frequently viewed in same session (proxy co-purchase)

Run:  uv run python scripts/ingest_retailrocket.py
"""
from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import track

console = Console(highlight=False, emoji=False)

DATA_DIR = Path("data/personalization/retailrocket")
DB_PATH  = Path("data/rufus_personalization.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS item_popularity (
    item_id   TEXT PRIMARY KEY,
    views     INTEGER DEFAULT 0,
    cart_adds INTEGER DEFAULT 0,
    purchases INTEGER DEFAULT 0,
    score     REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pop_score ON item_popularity(score DESC);

CREATE TABLE IF NOT EXISTS co_view (
    item_a TEXT,
    item_b TEXT,
    freq   INTEGER DEFAULT 1,
    PRIMARY KEY (item_a, item_b)
);
CREATE INDEX IF NOT EXISTS idx_coview_a ON co_view(item_a);
"""


def main() -> None:
    console.print("[bold]Ingest: RetailRocket -> rufus_personalization.db[/bold]")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    events_path = DATA_DIR / "events.csv"
    if not events_path.exists():
        console.print("[red]events.csv not found[/red]")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(_SCHEMA)
    conn.commit()

    console.print("Loading events.csv ...")
    df = pd.read_csv(events_path)
    console.print(f"  {len(df):,} events loaded")

    # ── Item popularity ───────────────────────────────────────────────────────
    console.print("Computing item popularity ...")
    views     = df[df["event"] == "view"].groupby("itemid").size()
    cart_adds = df[df["event"] == "addtocart"].groupby("itemid").size()
    purchases = df[df["event"] == "transaction"].groupby("itemid").size()

    all_items = set(views.index) | set(cart_adds.index) | set(purchases.index)
    rows = []
    for item in all_items:
        v = int(views.get(item, 0))
        c = int(cart_adds.get(item, 0))
        p = int(purchases.get(item, 0))
        score = round(p * 10 + c * 3 + v, 2)
        rows.append((str(item), v, c, p, score))

    conn.executemany(
        "INSERT OR REPLACE INTO item_popularity (item_id,views,cart_adds,purchases,score) "
        "VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    console.print(f"  item_popularity: {len(rows):,} items")

    # ── Co-view pairs (within same visitor session, view events only) ─────────
    console.print("Building co-view pairs ...")
    views_df = df[df["event"] == "view"][["visitorid", "itemid"]].dropna()
    # Group items per visitor
    visitor_items = views_df.groupby("visitorid")["itemid"].apply(list)

    pair_counts: Counter = Counter()
    for items in track(visitor_items, description="  Counting pairs..."):
        unique = list(dict.fromkeys(str(i) for i in items[:20]))  # cap at 20 per visitor
        for j in range(len(unique)):
            for k in range(j + 1, len(unique)):
                a, b = unique[j], unique[k]
                if a > b:
                    a, b = b, a
                pair_counts[(a, b)] += 1

    # Keep top 500K pairs by frequency
    top_pairs = pair_counts.most_common(500_000)
    conn.executemany(
        "INSERT OR REPLACE INTO co_view (item_a, item_b, freq) VALUES (?,?,?)",
        [(a, b, n) for (a, b), n in top_pairs],
    )
    conn.commit()
    console.print(f"  co_view: {len(top_pairs):,} pairs")

    total_pop = conn.execute("SELECT COUNT(*) FROM item_popularity").fetchone()[0]
    total_cov = conn.execute("SELECT COUNT(*) FROM co_view").fetchone()[0]
    conn.close()

    console.print(f"[bold green]Done.[/bold green]  "
                  f"item_popularity={total_pop:,}  co_view={total_cov:,}")


if __name__ == "__main__":
    main()
