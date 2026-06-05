#!/usr/bin/env python3
"""
Train demand forecasts at scale using statsforecast AutoETS + AutoARIMA ensemble.

Sources:
  M5 Walmart daily sales (30,490 SKUs x 1969 days)  -- primary
  Olist Brazilian e-commerce + DataCo + UCI Retail   -- from demand_history

Output:  rufus_sc.db forecasts table
  30-day ahead predictions for ALL SKUs with enough history (>= 14 days)
  Expected rows: ~900K (30K M5 + Olist/DataCo/UCI top SKUs)

Usage:
  uv run python scripts/train_demand_forecast.py          # all sources
  uv run python scripts/train_demand_forecast.py --m5-only
  uv run python scripts/train_demand_forecast.py --n-skus 1000  # quick test

Requires:  uv add statsforecast
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from rufus.inventory import get_db, init_db

console = Console(highlight=False, emoji=False)

M5_DIR  = Path("data/supply_chain/m5/m5/datasets")
HORIZON = 30
CHUNK   = 1_000   # SKUs per statsforecast batch

app = typer.Typer(help="Train demand forecasts at scale.")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _statsforecast_batch(df_long: pd.DataFrame, horizon: int) -> pd.DataFrame | None:
    """Run AutoETS + AutoARIMA on a batch; return forecast DataFrame or None."""
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS
        sf = StatsForecast(
            models=[AutoETS(season_length=7), AutoARIMA(season_length=7)],
            freq="D",
            n_jobs=-1,   # use all CPU cores
        )
        sf.fit(df_long)
        pred = sf.predict(h=horizon)
        # Average the two models for a simple ensemble
        pred["forecast"] = (pred["AutoETS"] + pred["AutoARIMA"]) / 2
        return pred[["unique_id", "ds", "forecast", "AutoETS", "AutoARIMA"]]
    except Exception as e:
        console.print(f"  [yellow]statsforecast failed: {e}  -> falling back to rolling avg[/yellow]")
        return None


def _rolling_forecast_batch(series_map: dict[str, pd.Series], horizon: int) -> list[tuple]:
    """Simple rolling average fallback — instant, no dependencies."""
    rows = []
    today = datetime.utcnow()
    for sku, series in series_map.items():
        arr = series.values[-30:].astype(float)
        arr = arr[np.isfinite(arr)]
        avg = float(arr.mean()) if len(arr) > 0 else 0.0
        trend = float(np.polyfit(range(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0
        ci = float(arr.std() * 1.96) if len(arr) > 1 else avg * 0.2
        for i in range(horizon):
            pred = max(avg + trend * i, 0.0)
            fdate = (today + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            rows.append((sku, fdate, round(pred, 2),
                         max(round(pred - ci, 2), 0.0), round(pred + ci, 2)))
    return rows


def _build_long_df(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Convert {sku: pd.Series(date_index)} to statsforecast long format."""
    frames = []
    for sku, s in series_map.items():
        tmp = pd.DataFrame({"unique_id": sku, "ds": pd.to_datetime(s.index), "y": s.values.astype(float)})
        tmp = tmp[np.isfinite(tmp["y"])]
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _forecast_and_store(series_map: dict[str, pd.Series], horizon: int) -> int:
    """Run forecast on series_map, write to DB, return number of rows stored."""
    df_long = _build_long_df(series_map)
    today   = datetime.utcnow()
    rows: list[tuple] = []

    pred_df = _statsforecast_batch(df_long, horizon) if not df_long.empty else None

    if pred_df is not None:
        for _, r in pred_df.iterrows():
            sku   = r["unique_id"]
            fdate = r["ds"].strftime("%Y-%m-%d")
            pred  = max(float(r["forecast"]), 0.0)
            lo    = max(float(r.get("AutoETS", pred) * 0.8), 0.0)
            hi    = float(r.get("AutoARIMA", pred) * 1.2)
            rows.append((sku, fdate, round(pred, 2), round(lo, 2), round(hi, 2)))
    else:
        rows = _rolling_forecast_batch(series_map, horizon)

    with get_db() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO forecasts "
            "(sku, forecast_date, predicted_units, lower_ci, upper_ci) VALUES (?,?,?,?,?)",
            rows,
        )
    return len(rows)


# ── M5 full-scale ─────────────────────────────────────────────────────────────

def train_m5(n_skus: int = 0) -> None:
    sales_path = M5_DIR / "sales_train_evaluation.csv"
    cal_path   = M5_DIR / "calendar.csv"
    if not sales_path.exists():
        console.print(f"  [yellow]M5 not found at {sales_path} — skipping[/yellow]")
        return

    console.print("[bold]M5 full-scale demand forecast[/bold]")
    console.print("  Loading M5 sales matrix ...")
    df  = pd.read_csv(sales_path)
    cal = pd.read_csv(cal_path, usecols=["date"])

    id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]

    # Map d_N column -> calendar date
    d2date = {f"d_{i+1}": row for i, row in enumerate(cal["date"])}

    if n_skus > 0:
        df["_total"] = df[day_cols].sum(axis=1)
        df = df.nlargest(n_skus, "_total")
    total_skus = len(df)
    console.print(f"  {total_skus:,} SKUs to forecast")

    # Melt the wide matrix into a dict of series
    console.print("  Melting M5 sales matrix (this takes ~30s) ...")
    df_melt = df[id_cols + day_cols].melt(
        id_vars=["item_id", "store_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="units",
    )
    df_melt["date"]  = df_melt["d"].map(d2date)
    df_melt["sku"]   = df_melt["item_id"] + "_" + df_melt["store_id"]
    df_melt          = df_melt.dropna(subset=["date"])
    df_melt["date"]  = pd.to_datetime(df_melt["date"])
    df_melt          = df_melt.sort_values(["sku", "date"])

    # Store full demand history first
    console.print("  Updating demand_history (M5) ...")
    hist_rows = [
        (r.sku, r.date.strftime("%Y-%m-%d"), int(r.units), "m5_walmart")
        for r in df_melt.itertuples()
    ]
    with get_db() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO demand_history (sku, date, units_sold, store_id) VALUES (?,?,?,?)",
            hist_rows,
        )
    console.print(f"  demand_history (M5): {len(hist_rows):,} rows")

    # Forecast in chunks
    skus   = df_melt["sku"].unique().tolist()
    total_fc = 0
    with Progress(SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as prog:
        task = prog.add_task("  Forecasting M5 ...", total=len(skus))
        for i in range(0, len(skus), CHUNK):
            batch_skus = skus[i : i + CHUNK]
            series_map: dict[str, pd.Series] = {}
            for sku in batch_skus:
                s = df_melt[df_melt["sku"] == sku].set_index("date")["units"]
                if len(s) >= 14:
                    series_map[sku] = s
            if series_map:
                total_fc += _forecast_and_store(series_map, HORIZON)
            prog.advance(task, len(batch_skus))

    console.print(f"  [green]M5 forecast rows stored: {total_fc:,}[/green]")


# ── History-based (Olist + DataCo + UCI) ─────────────────────────────────────

def train_from_history(n_skus: int = 0) -> None:
    console.print("[bold]Forecasting from demand_history (Olist + DataCo + UCI)[/bold]")

    with get_db() as conn:
        rows = conn.execute(
            "SELECT sku, date, SUM(units_sold) as u "
            "FROM demand_history WHERE store_id != 'm5_walmart' "
            "GROUP BY sku, date ORDER BY sku, date"
        ).fetchall()

    if not rows:
        console.print("  No non-M5 demand history found")
        return

    df = pd.DataFrame(rows, columns=["sku", "date", "units"])
    df["date"] = pd.to_datetime(df["date"])

    top_skus = df.groupby("sku")["units"].sum().nlargest(n_skus or len(df["sku"].unique())).index.tolist()
    console.print(f"  {len(top_skus):,} SKUs to forecast")

    total_fc = 0
    with Progress(SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as prog:
        task = prog.add_task("  Forecasting ...", total=len(top_skus))
        for i in range(0, len(top_skus), CHUNK):
            batch_skus = top_skus[i : i + CHUNK]
            series_map: dict[str, pd.Series] = {}
            for sku in batch_skus:
                s = df[df["sku"] == sku].set_index("date")["units"]
                if len(s) >= 7:
                    series_map[sku] = s
            if series_map:
                total_fc += _forecast_and_store(series_map, HORIZON)
            prog.advance(task, len(batch_skus))

    console.print(f"  [green]history forecast rows stored: {total_fc:,}[/green]")


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    m5_only:  bool = typer.Option(False,  "--m5-only",  help="Only forecast M5 series"),
    n_skus:   int  = typer.Option(0,      "--n-skus",   help="Limit SKUs (0=all)"),
) -> None:
    init_db()

    if not m5_only:
        train_from_history(n_skus)

    train_m5(n_skus)

    with get_db() as conn:
        n = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        skus = conn.execute("SELECT COUNT(DISTINCT sku) FROM forecasts").fetchone()[0]
    console.print(f"\n[bold green]Forecast training complete.[/bold green]  "
                  f"{skus:,} SKUs  x  {HORIZON} days  =  {n:,} forecast rows")


if __name__ == "__main__":
    app()
