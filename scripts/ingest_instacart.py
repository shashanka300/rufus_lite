#!/usr/bin/env python3
"""
Ingest Instacart basket data into rufus_personalization.db.

Source: data/personalization/instacart/
  order_products__prior.csv  ~32M rows (order_id, product_id, ...)
  order_products__train.csv  ~1.4M rows
  aisles.csv                 aisle_id -> aisle name
  departments.csv            department_id -> name

Populates rufus_personalization.db:
  basket_copurchase  -- product pairs frequently bought together
  product_popularity -- total purchases per product_id

These are Instacart grocery IDs (not ASINs), but co-purchase patterns
are used as a demonstration layer for "often bought together" features.

Run:  uv run python scripts/ingest_instacart.py
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import track

console = Console(highlight=False, emoji=False)

DATA_DIR = Path("data/personalization/instacart")
DB_PATH  = Path("data/rufus_personalization.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS basket_copurchase (
    product_a TEXT,
    product_b TEXT,
    freq      INTEGER DEFAULT 1,
    PRIMARY KEY (product_a, product_b)
);
CREATE INDEX IF NOT EXISTS idx_bcp_a ON basket_copurchase(product_a);

CREATE TABLE IF NOT EXISTS product_popularity (
    product_id   TEXT PRIMARY KEY,
    order_count  INTEGER DEFAULT 0,
    reorder_rate REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pp_count ON product_popularity(order_count DESC);
"""

MAX_ITEMS_PER_ORDER = 15   # skip very large orders (>15 items) to control pairs explosion
TOP_PAIRS = 1_000_000      # keep top 1M co-purchase pairs


def _build_copurchase(path: Path, pair_counts: Counter, pop: Counter) -> int:
    console.print(f"  Reading {path.name} ...")
    df = pd.read_csv(path, usecols=["order_id", "product_id", "reordered"])
    console.print(f"    {len(df):,} rows")

    # Count product popularity
    pop.update(df["product_id"].values)

    # Group products per order
    grouped = df.groupby("order_id")["product_id"].apply(list)
    total_orders = 0

    for items in track(grouped, description=f"    Counting pairs ({path.name})..."):
        if len(items) > MAX_ITEMS_PER_ORDER:
            continue
        total_orders += 1
        unique = list(dict.fromkeys(str(i) for i in items))
        for j in range(len(unique)):
            for k in range(j + 1, len(unique)):
                a, b = unique[j], unique[k]
                if a > b:
                    a, b = b, a
                pair_counts[(a, b)] += 1

    return total_orders


def main() -> None:
    console.print("[bold]Ingest: Instacart -> rufus_personalization.db[/bold]")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(_SCHEMA)
    conn.commit()

    pair_counts: Counter = Counter()
    pop: Counter = Counter()
    total = 0

    for fname in ("order_products__prior.csv", "order_products__train.csv"):
        p = DATA_DIR / fname
        if p.exists():
            total += _build_copurchase(p, pair_counts, pop)

    # ── Write co-purchase pairs ───────────────────────────────────────────────
    top = pair_counts.most_common(TOP_PAIRS)
    console.print(f"Writing {len(top):,} co-purchase pairs ...")
    conn.executemany(
        "INSERT OR REPLACE INTO basket_copurchase (product_a, product_b, freq) VALUES (?,?,?)",
        [(a, b, n) for (a, b), n in top],
    )
    conn.commit()

    # ── Write product popularity ──────────────────────────────────────────────
    console.print(f"Writing {len(pop):,} product popularity entries ...")
    pop_rows = [(str(pid), int(cnt), 0.0) for pid, cnt in pop.most_common()]
    conn.executemany(
        "INSERT OR REPLACE INTO product_popularity (product_id, order_count, reorder_rate) "
        "VALUES (?,?,?)",
        pop_rows,
    )
    conn.commit()
    conn.close()

    console.print(f"[bold green]Done.[/bold green]  "
                  f"orders processed={total:,}  "
                  f"pairs={len(top):,}  products={len(pop):,}")


if __name__ == "__main__":
    main()
