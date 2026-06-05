#!/usr/bin/env python3
"""
Ingest UCI Online Retail II into rufus_sc.db demand_history.

Source: data/supply_chain/uci_retail/online_retail_ii.zip
  ~1M UK e-commerce transactions 2009-2011.
  Columns: Invoice, StockCode, Description, Quantity, InvoiceDate,
           Price, Customer ID, Country

Adds daily demand per StockCode to rufus_sc.db demand_history and
populates inventory for items not already present.

Run:  uv run python scripts/ingest_uci_retail.py
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd
from rich.console import Console

from rufus.inventory import get_db, init_db

console = Console(highlight=False, emoji=False)

XLSX_PATH = Path("data/supply_chain/uci_retail/online_retail_II.xlsx")
DB_PATH   = Path("data/rufus_sc.db")


def _read_source() -> pd.DataFrame:
    if XLSX_PATH.exists():
        console.print(f"  reading {XLSX_PATH.name} ...")
        frames = []
        xl = pd.ExcelFile(XLSX_PATH, engine="openpyxl")
        for sheet in xl.sheet_names:
            console.print(f"    sheet: {sheet}")
            frames.append(xl.parse(sheet))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    console.print(f"[red]{XLSX_PATH} not found[/red]")
    return pd.DataFrame()


def main() -> None:
    console.print("[bold]Ingest: UCI Retail II -> rufus_sc.db[/bold]")
    init_db()

    if not XLSX_PATH.exists():
        console.print(f"[red]{ZIP_PATH} not found[/red]")
        return

    # Check if already ingested
    with get_db() as conn:
        existing = conn.execute(
            "SELECT COUNT(*) FROM demand_history WHERE store_id='uci_retail'"
        ).fetchone()[0]
    if existing > 100_000:
        console.print(f"  already ingested ({existing:,} rows), skipping")
        return

    df = _read_source()
    console.print(f"  loaded {len(df):,} rows")
    if df.empty:
        return

    # Normalise column names (handle both possible cases)
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    invoice_date_col = col_map.get("invoicedate") or col_map.get("invoice date")
    stock_code_col   = col_map.get("stockcode")   or col_map.get("stock code")
    quantity_col     = col_map.get("quantity")
    price_col        = col_map.get("price")        or col_map.get("unit price")
    desc_col         = col_map.get("description")

    if not all([invoice_date_col, stock_code_col, quantity_col]):
        console.print(f"[red]Unexpected columns: {df.columns.tolist()}[/red]")
        return

    df["_date"] = pd.to_datetime(df[invoice_date_col], errors="coerce")
    df["_qty"]  = pd.to_numeric(df[quantity_col], errors="coerce").fillna(0)
    df["_code"] = df[stock_code_col].astype(str).str.strip()
    df["_price"] = pd.to_numeric(df[price_col], errors="coerce").fillna(0) if price_col else 0

    # Keep positive quantities only (returns are negative)
    df = df[(df["_qty"] > 0) & df["_date"].notna() & (df["_code"] != "")]
    df["_date_str"] = df["_date"].dt.strftime("%Y-%m-%d")
    console.print(f"  after cleaning: {len(df):,} rows")

    # Rename internal cols so itertuples works (no leading underscore)
    df = df.rename(columns={"_code": "code", "_date_str": "date_str", "_qty": "qty", "_price": "price_val"})

    # ── Demand history ────────────────────────────────────────────────────────
    daily = df.groupby(["code", "date_str"])["qty"].sum().reset_index()
    dem_rows = [
        (hashlib.sha256(r.code.encode()).hexdigest()[:16],
         r.date_str, int(r.qty), "uci_retail")
        for r in daily.itertuples()
    ]
    with get_db() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO demand_history (sku, date, units_sold, store_id) "
            "VALUES (?,?,?,?)",
            dem_rows,
        )
    console.print(f"  demand_history: +{len(dem_rows):,} rows")

    # ── Inventory (new SKUs only) ─────────────────────────────────────────────
    prod_agg = df.groupby("code").agg(
        total_qty = ("qty", "sum"),
        avg_price = ("price_val", "mean"),
        title     = (desc_col if desc_col in df.columns else "code", "first"),
    ).reset_index()

    import numpy as np
    rng = np.random.default_rng(99)
    n   = len(prod_agg)
    rp  = (prod_agg["total_qty"] * 0.15).clip(lower=5).astype(int).values
    oh  = (rng.uniform(0.5, 2.5, n) * rp).astype(int)
    now = pd.Timestamp.utcnow().isoformat()

    inv_rows = []
    for i, r in enumerate(prod_agg.itertuples()):
        sku = hashlib.sha256(r.code.encode()).hexdigest()[:16]
        inv_rows.append((sku, str(r.title)[:80], "retail", int(oh[i]),
                         int(rp[i]), max(int(rp[i] * 0.3), 1),
                         round(float(r.avg_price) * 0.6, 2),
                         round(float(r.avg_price), 2), 7, now))

    with get_db() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO inventory "
            "(sku,title,category,qty_on_hand,reorder_pt,safety_stock,"
            "unit_cost,unit_price,lead_time,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            inv_rows,
        )
    console.print(f"  inventory: +{len(inv_rows):,} SKUs (new only)")

    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
