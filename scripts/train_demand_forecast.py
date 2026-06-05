#!/usr/bin/env python3
"""
Train demand forecasts from M5 + Olist demand_history and store in rufus_sc.db.

Uses statsforecast AutoETS (much faster than Prophet, comparable accuracy).
Falls back to rolling-average if statsforecast not installed.

Run after: scripts/ingest_olist.py  AND  data/supply_chain/m5 is present

Output: forecasts table in rufus_sc.db — 30-day ahead predictions per SKU.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track

from rufus.inventory import DB_PATH, get_db, init_db

console = Console(highlight=False, emoji=False)
M5_DIR  = Path("data/supply_chain/m5/m5/datasets")
HORIZON = 30   # days ahead to forecast
TOP_N   = 500  # forecast top-N highest-demand SKUs (full 32K would take hours)


def _rolling_forecast(series: pd.Series, horizon: int = HORIZON) -> list[float]:
    """Simple seasonal naive + trend: avg last 30 days, adjusted for weekly pattern."""
    if len(series) < 7:
        avg = float(series.mean()) if len(series) > 0 else 0
        return [max(avg, 0)] * horizon
    window = series.tail(30)
    avg = float(window.mean())
    std = float(window.std()) if len(window) > 1 else 0
    # Add slight trend component
    trend = float(np.polyfit(range(len(window)), window.values, 1)[0])
    preds = [max(avg + trend * i + np.random.normal(0, std * 0.1), 0) for i in range(horizon)]
    return preds


def _try_statsforecast(series: pd.Series, horizon: int) -> list[float] | None:
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS
        df = pd.DataFrame({
            "unique_id": ["sku"],
            "ds": pd.date_range(end=datetime.utcnow(), periods=len(series), freq="D"),
            "y": series.values.astype(float),
        })
        sf = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=1)
        sf.fit(df)
        pred = sf.predict(h=horizon)
        return pred["AutoETS"].tolist()
    except Exception:
        return None


def train_from_olist() -> None:
    """Train forecasts for top SKUs in Olist demand_history."""
    console.print("[bold]Training demand forecasts from Olist history ...[/bold]")

    with get_db() as conn:
        # Get top-N SKUs by total demand
        rows = conn.execute(
            """SELECT sku, date, SUM(units_sold) as units
               FROM demand_history
               GROUP BY sku, date
               ORDER BY sku, date"""
        ).fetchall()

    if not rows:
        console.print("  [yellow]No demand_history data. Run ingest_olist.py first.[/yellow]")
        return

    df = pd.DataFrame(rows, columns=["sku", "date", "units"])
    top_skus = (df.groupby("sku")["units"].sum()
                  .nlargest(TOP_N).index.tolist())

    console.print(f"  Forecasting {len(top_skus)} top SKUs ...")
    forecast_rows = []
    today = datetime.utcnow().strftime("%Y-%m-%d")

    for sku in track(top_skus, description="Forecasting..."):
        series = df[df["sku"] == sku].set_index("date")["units"].sort_index()
        preds = _try_statsforecast(series, HORIZON) or _rolling_forecast(series, HORIZON)
        ci    = np.std(preds) * 1.96
        for i, pred in enumerate(preds):
            fdate = (datetime.utcnow() + timedelta(days=i+1)).strftime("%Y-%m-%d")
            forecast_rows.append((sku, fdate, round(pred, 2),
                                   max(round(pred - ci, 2), 0), round(pred + ci, 2)))

    with get_db() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO forecasts
               (sku, forecast_date, predicted_units, lower_ci, upper_ci)
               VALUES (?,?,?,?,?)""",
            forecast_rows,
        )
    console.print(f"  [green]inserted {len(forecast_rows):,} forecast rows[/green]")


def train_from_m5() -> None:
    """Load M5 Walmart daily sales and insert into demand_history for richer signals."""
    sales_path = M5_DIR / "sales_train_evaluation.csv"
    if not sales_path.exists():
        console.print(f"  [yellow]M5 not found at {sales_path} -- skipping[/yellow]")
        return

    console.print("[bold]Loading M5 Walmart demand history ...[/bold]")
    df = pd.read_csv(sales_path)
    # datasetsforecast M5 calendar has no 'd' column — generate d_1..d_N from row index
    cal = pd.read_csv(M5_DIR / "calendar.csv", usecols=["date"])
    d2date = {f"d_{i+1}": row for i, row in enumerate(cal["date"])}

    # Take top-500 items by total sales
    id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]
    df["total"] = df[day_cols].sum(axis=1)
    top = df.nlargest(500, "total")

    rows = []
    for _, row in track(top.iterrows(), total=len(top), description="M5 -> demand_history ..."):
        sku = row["item_id"] + "_" + row["store_id"]
        for d in day_cols[-90:]:  # last 90 days only to keep DB small
            date = d2date.get(d)
            if date:
                rows.append((sku, date, int(row[d]), "m5_walmart"))

    with get_db() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO demand_history (sku, date, units_sold, store_id)
               VALUES (?,?,?,?)""",
            rows,
        )
    console.print(f"  [green]inserted {len(rows):,} M5 demand rows[/green]")


def main() -> None:
    init_db()
    train_from_m5()
    train_from_olist()
    console.print("[bold green]Demand forecast training complete.[/bold green]")
    with get_db() as conn:
        n = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        console.print(f"  forecasts table: {n:,} rows")


if __name__ == "__main__":
    main()
