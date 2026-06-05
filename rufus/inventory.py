"""
Supply chain inventory layer — SQLite-backed.

Tables
------
  inventory       current stock levels per SKU
  demand_history  daily units sold per SKU
  forecasts       pre-computed demand forecasts
  suppliers       vendor catalog with lead times
  purchase_orders open/closed POs
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("data/rufus_sc.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inventory (
    sku          TEXT PRIMARY KEY,
    title        TEXT,
    category     TEXT,
    qty_on_hand  INTEGER DEFAULT 0,
    reorder_pt   INTEGER DEFAULT 0,
    safety_stock INTEGER DEFAULT 0,
    unit_cost    REAL    DEFAULT 0,
    unit_price   REAL    DEFAULT 0,
    lead_time    INTEGER DEFAULT 14,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS demand_history (
    sku        TEXT,
    date       TEXT,
    units_sold INTEGER DEFAULT 0,
    store_id   TEXT    DEFAULT 'default',
    PRIMARY KEY (sku, date, store_id)
);

CREATE TABLE IF NOT EXISTS forecasts (
    sku             TEXT,
    forecast_date   TEXT,
    predicted_units REAL,
    lower_ci        REAL,
    upper_ci        REAL,
    PRIMARY KEY (sku, forecast_date)
);

CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id    TEXT PRIMARY KEY,
    name           TEXT,
    country        TEXT,
    lead_time_days INTEGER DEFAULT 14,
    freight_mode   TEXT,
    avg_unit_price REAL    DEFAULT 0
);

CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id         TEXT PRIMARY KEY,
    sku           TEXT,
    supplier_id   TEXT,
    qty_ordered   INTEGER,
    unit_price    REAL,
    order_date    TEXT,
    expected_date TEXT,
    status        TEXT DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_demand_sku  ON demand_history(sku);
CREATE INDEX IF NOT EXISTS idx_demand_date ON demand_history(date);
CREATE INDEX IF NOT EXISTS idx_forecast_sku ON forecasts(sku);
CREATE INDEX IF NOT EXISTS idx_inv_cat ON inventory(category);
"""


def get_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(path: Path = DB_PATH) -> None:
    with get_db(path) as conn:
        conn.executescript(_SCHEMA)


# ── Inventory ─────────────────────────────────────────────────────────────────

def get_item(sku: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM inventory WHERE sku = ?", (sku,)
        ).fetchone()
        return dict(row) if row else None


def search_inventory(query: str, limit: int = 10) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM inventory
               WHERE lower(title) LIKE lower(?) OR lower(category) LIKE lower(?)
               ORDER BY qty_on_hand DESC LIMIT ?""",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_low_stock(limit: int = 20) -> list[dict]:
    """Items where qty_on_hand < reorder_pt, ordered by shortage severity."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT *, (reorder_pt - qty_on_hand) AS shortage
               FROM inventory
               WHERE qty_on_hand < reorder_pt AND reorder_pt > 0
               ORDER BY shortage DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_out_of_stock(limit: int = 20) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM inventory WHERE qty_on_hand <= 0 LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_overstock(ratio: float = 3.0, limit: int = 20) -> list[dict]:
    """Items with qty_on_hand > ratio * reorder_pt (excess inventory)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT *, (qty_on_hand - reorder_pt) AS excess
               FROM inventory
               WHERE reorder_pt > 0 AND qty_on_hand > reorder_pt * ?
               ORDER BY excess DESC LIMIT ?""",
            (ratio, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def summary_stats() -> dict:
    with get_db() as conn:
        r = conn.execute("""
            SELECT
                COUNT(*) as total_skus,
                SUM(CASE WHEN qty_on_hand <= 0 THEN 1 ELSE 0 END) as out_of_stock,
                SUM(CASE WHEN qty_on_hand < reorder_pt THEN 1 ELSE 0 END) as low_stock,
                SUM(qty_on_hand) as total_units,
                ROUND(SUM(qty_on_hand * unit_cost), 2) as inventory_value
            FROM inventory
        """).fetchone()
        return dict(r)


# ── Demand history ────────────────────────────────────────────────────────────

def get_recent_demand(sku: str, days: int = 90) -> list[dict]:
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_db() as conn:
        rows = conn.execute(
            """SELECT date, SUM(units_sold) as units
               FROM demand_history WHERE sku = ? AND date >= ?
               GROUP BY date ORDER BY date""",
            (sku, cutoff),
        ).fetchall()
        return [dict(r) for r in rows]


def avg_daily_demand(sku: str, days: int = 30) -> float:
    history = get_recent_demand(sku, days)
    if not history:
        return 0.0
    total = sum(r["units"] for r in history)
    return total / max(len(history), 1)


# ── Forecasting ───────────────────────────────────────────────────────────────

def get_forecast(sku: str, horizon_days: int = 30) -> dict:
    """
    Return demand forecast for next horizon_days.
    Uses pre-computed Prophet forecasts when available; falls back to
    rolling 30-day average.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with get_db() as conn:
        rows = conn.execute(
            """SELECT forecast_date, predicted_units, lower_ci, upper_ci
               FROM forecasts WHERE sku = ? AND forecast_date >= ?
               ORDER BY forecast_date LIMIT ?""",
            (sku, today, horizon_days),
        ).fetchall()

    if rows:
        total = sum(r["predicted_units"] for r in rows)
        return {
            "sku": sku,
            "horizon_days": len(rows),
            "total_predicted": round(total, 1),
            "daily_avg": round(total / len(rows), 2),
            "source": "prophet",
        }

    avg = avg_daily_demand(sku, days=30)
    return {
        "sku": sku,
        "horizon_days": horizon_days,
        "total_predicted": round(avg * horizon_days, 1),
        "daily_avg": round(avg, 2),
        "source": "rolling_avg",
    }


def days_of_stock(sku: str) -> float | None:
    item = get_item(sku)
    if not item:
        return None
    daily = avg_daily_demand(sku) or 1.0
    return round(item["qty_on_hand"] / daily, 1)


def reorder_recommendation(sku: str) -> dict | None:
    item = get_item(sku)
    if not item:
        return None
    avg = avg_daily_demand(sku) or 0.1
    lead = item.get("lead_time") or 14
    dos = round(item["qty_on_hand"] / avg, 1)
    urgent = item["qty_on_hand"] < item["reorder_pt"]
    order_qty = max(int(avg * 30), item.get("safety_stock") or 1)
    return {
        "sku": sku,
        "title": item["title"],
        "category": item["category"],
        "qty_on_hand": item["qty_on_hand"],
        "reorder_pt": item["reorder_pt"],
        "days_of_stock": dos,
        "avg_daily_demand": round(avg, 2),
        "lead_time_days": lead,
        "urgent": urgent,
        "recommended_order_qty": order_qty,
        "estimated_cost": round(order_qty * (item.get("unit_cost") or 0), 2),
    }


# ── Supplier queries ──────────────────────────────────────────────────────────

def get_all_suppliers(limit: int = 20) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM suppliers ORDER BY lead_time_days ASC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_supplier(supplier_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM suppliers WHERE supplier_id = ?", (supplier_id,)
        ).fetchone()
        return dict(row) if row else None


def get_open_pos(limit: int = 20) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT po.*, s.name as supplier_name, s.lead_time_days,
                      i.title as product_title
               FROM purchase_orders po
               LEFT JOIN suppliers s ON s.supplier_id = po.supplier_id
               LEFT JOIN inventory i ON i.sku = po.sku
               WHERE po.status = 'pending'
               ORDER BY po.expected_date ASC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
