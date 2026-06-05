#!/usr/bin/env python3
"""
Ingest Olist Brazilian E-Commerce data into rufus_sc.db.

Populates:
  inventory       -- products with simulated stock levels
  demand_history  -- daily units sold per product (from order timestamps)
  suppliers       -- sellers as proxy vendors
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console

from rufus.inventory import DB_PATH, get_db, init_db

console = Console(highlight=False, emoji=False)
DATA_DIR = Path("data/supply_chain/olist")


def main() -> None:
    console.print("[bold]Ingest: Olist -> rufus_sc.db[/bold]")
    init_db()

    console.print("Loading CSVs ...")
    products   = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
    xlat       = pd.read_csv(DATA_DIR / "product_category_name_translation.csv")
    orders     = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv",
                             parse_dates=["order_purchase_timestamp"])
    items      = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
    sellers    = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")

    products = products.merge(xlat, on="product_category_name", how="left")
    products["category"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown")
    )

    # ── Inventory ─────────────────────────────────────────────────────────────
    console.print("Building inventory ...")
    units_sold = (items.groupby("product_id")["order_item_id"]
                  .count().rename("total_sold").reset_index())
    avg_price  = (items.groupby("product_id")["price"]
                  .mean().rename("avg_price").reset_index())

    inv = (products
           .merge(units_sold, on="product_id", how="left")
           .merge(avg_price,  on="product_id", how="left"))
    inv["total_sold"] = inv["total_sold"].fillna(0).astype(int)
    inv["avg_price"]  = inv["avg_price"].fillna(0.0)

    # Simulate realistic stock distribution.
    # 85% of products have <5 historical sales (long tail), so we can't use
    # pure multipliers — use absolute reorder points + seeded random on-hand.
    import numpy as np
    rng = np.random.default_rng(42)
    n = len(inv)

    # Reorder point: max(5% of historical sales, 10 units minimum)
    inv["reorder_pt"]   = (inv["total_sold"] * 0.15).clip(lower=10).astype(int)
    inv["safety_stock"] = (inv["total_sold"] * 0.05).clip(lower=5).astype(int)

    # On-hand: draw from 5 stock scenarios with realistic proportions
    #   5% out of stock, 15% critical (<reorder), 20% low, 40% healthy, 20% over
    scenario = rng.choice(5, size=n, p=[0.05, 0.15, 0.20, 0.40, 0.20])
    rp = inv["reorder_pt"].values
    oh = np.where(scenario == 0, 0,
         np.where(scenario == 1, (rng.uniform(0.01, 0.5, n) * rp).astype(int),
         np.where(scenario == 2, (rng.uniform(0.5, 0.95, n) * rp).astype(int),
         np.where(scenario == 3, (rng.uniform(1.0, 2.5, n) * rp).astype(int),
                                 (rng.uniform(2.5, 5.0, n) * rp).astype(int)))))
    inv["qty_on_hand"] = oh.astype(int)

    now = datetime.utcnow().isoformat()
    inv_rows = [
        (
            row.product_id,
            f"{row.category} — {row.product_id[:8]}",
            row.category,
            int(row.qty_on_hand),
            int(row.reorder_pt),
            int(row.safety_stock),
            round(float(row.avg_price) * 0.6, 2),
            round(float(row.avg_price), 2),
            14,
            now,
        )
        for row in inv.itertuples()
    ]
    with get_db() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO inventory
               (sku, title, category, qty_on_hand, reorder_pt, safety_stock,
                unit_cost, unit_price, lead_time, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            inv_rows,
        )
    console.print(f"  inventory: {len(inv_rows):,} rows")

    # ── Demand history ────────────────────────────────────────────────────────
    console.print("Building demand history ...")
    merged = items.merge(
        orders[["order_id", "order_purchase_timestamp"]].dropna(),
        on="order_id", how="inner",
    )
    merged["date"] = merged["order_purchase_timestamp"].dt.strftime("%Y-%m-%d")
    daily = (merged.groupby(["product_id", "date"])["order_item_id"]
             .count().reset_index())
    daily.columns = ["sku", "date", "units_sold"]

    demand_rows = [(r.sku, r.date, int(r.units_sold), "olist")
                   for r in daily.itertuples()]
    with get_db() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO demand_history (sku, date, units_sold, store_id)
               VALUES (?,?,?,?)""",
            demand_rows,
        )
    console.print(f"  demand_history: {len(demand_rows):,} rows")

    # ── Suppliers (from sellers) ──────────────────────────────────────────────
    console.print("Building supplier table from sellers ...")
    seller_stats = (items.groupby("seller_id")["price"]
                    .mean().rename("avg_price").reset_index())
    sellers = sellers.merge(seller_stats, on="seller_id", how="left")
    sellers["avg_price"] = sellers["avg_price"].fillna(0.0)

    supplier_rows = [
        (
            row.seller_id,
            f"Seller {row.seller_id[:8]} ({row.seller_city})",
            "Brazil",
            7,
            "standard",
            round(float(row.avg_price), 2),
        )
        for row in sellers.itertuples()
    ]
    with get_db() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO suppliers
               (supplier_id, name, country, lead_time_days, freight_mode, avg_unit_price)
               VALUES (?,?,?,?,?,?)""",
            supplier_rows,
        )
    console.print(f"  suppliers: {len(supplier_rows):,} rows")

    # Final counts
    console.print("\n[bold green]Olist ingest complete.[/bold green]")
    with get_db() as conn:
        for tbl in ("inventory", "demand_history", "suppliers"):
            n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            console.print(f"  {tbl}: {n:,}")


if __name__ == "__main__":
    main()
