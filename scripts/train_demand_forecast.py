#!/usr/bin/env python3
"""
Train demand forecasts on GPU using NeuralForecast NHITS.

NeuralForecast trains ONE global model across ALL series simultaneously on GPU,
vs statsforecast which fits one CPU model per series (orders of magnitude slower).

RTX 5090 / 32GB VRAM — all 30K M5 series fit comfortably in a single GPU pass.

Sources:
  M5 Walmart daily sales (30,490 SKUs x 1969 days)   primary
  Olist + DataCo + UCI Retail from demand_history     secondary

Output: rufus_sc.db forecasts table
  30-day ahead per SKU, ~900K rows total

Usage:
  uv run python scripts/train_demand_forecast.py          # all sources
  uv run python scripts/train_demand_forecast.py --m5-only
  uv run python scripts/train_demand_forecast.py --n-skus 500  # quick test

Requires: uv add neuralforecast
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn

from rufus.inventory import get_db, init_db

console = Console(highlight=False, emoji=False)

M5_DIR  = Path("data/supply_chain/m5/m5/datasets")
HORIZON = 30
MIN_LEN = 30   # minimum history length to include a series

app = typer.Typer(help="Train demand forecasts on GPU (NeuralForecast NHITS).")


# ── GPU forecast ──────────────────────────────────────────────────────────────

def _gpu_forecast(df_long: pd.DataFrame, horizon: int) -> pd.DataFrame | None:
    """
    Run NHITS on GPU across all series at once.
    Returns a long-format DataFrame with columns [unique_id, ds, NHITS].
    """
    try:
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS

        device = "gpu" if torch.cuda.is_available() else "cpu"
        console.print(f"  [cyan]NeuralForecast NHITS on {device.upper()}[/cyan]  "
                      f"({len(df_long['unique_id'].unique()):,} series)")

        model = NHITS(
            h=horizon,
            input_size=horizon * 4,   # look-back window = 4x horizon
            max_steps=500,
            accelerator=device,
            devices=1,
            enable_progress_bar=True,
            logger=False,
        )
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df_long)
        pred = nf.predict().reset_index()
        pred.columns = [c.replace("NHITS", "forecast") if c == "NHITS" else c
                        for c in pred.columns]
        return pred

    except Exception as e:
        console.print(f"  [yellow]NeuralForecast failed: {e}[/yellow]")
        return None


def _rolling_fallback(df_long: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fast rolling-average fallback when GPU is unavailable."""
    records = []
    today = pd.Timestamp.utcnow().normalize()
    for uid, grp in df_long.groupby("unique_id"):
        arr = grp.sort_values("ds")["y"].values[-30:].astype(float)
        arr = arr[np.isfinite(arr)]
        avg = float(arr.mean()) if len(arr) > 0 else 0.0
        for i in range(horizon):
            records.append({"unique_id": uid,
                             "ds": today + timedelta(days=i + 1),
                             "forecast": max(avg, 0.0)})
    return pd.DataFrame(records)


def _store(pred: pd.DataFrame) -> int:
    """Write forecast DataFrame to rufus_sc.db forecasts table."""
    today = datetime.utcnow()
    rows = []
    for r in pred.itertuples():
        pred_val = max(float(getattr(r, "forecast", 0)), 0.0)
        ci = pred_val * 0.15
        rows.append((
            str(r.unique_id),
            pd.Timestamp(r.ds).strftime("%Y-%m-%d"),
            round(pred_val, 2),
            round(max(pred_val - ci, 0.0), 2),
            round(pred_val + ci, 2),
        ))
    with get_db() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO forecasts "
            "(sku, forecast_date, predicted_units, lower_ci, upper_ci) VALUES (?,?,?,?,?)",
            rows,
        )
    return len(rows)


def _already_forecasted() -> set[str]:
    with get_db() as conn:
        return {r[0] for r in conn.execute("SELECT DISTINCT sku FROM forecasts").fetchall()}


# ── M5 ────────────────────────────────────────────────────────────────────────

def train_m5(n_skus: int = 0) -> None:
    sales_path = M5_DIR / "sales_train_evaluation.csv"
    cal_path   = M5_DIR / "calendar.csv"
    if not sales_path.exists():
        console.print(f"  [yellow]M5 not found — skipping[/yellow]")
        return

    console.print("[bold]M5 — loading data ...[/bold]")
    df  = pd.read_csv(sales_path)
    cal = pd.read_csv(cal_path, usecols=["date"])

    id_cols  = [c for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"] if c in df.columns]
    day_cols = [c for c in df.columns if c.startswith("d_")]
    d2date   = {f"d_{i+1}": d for i, d in enumerate(cal["date"])}

    if n_skus > 0:
        df["_tot"] = df[day_cols].sum(axis=1)
        df = df.nlargest(n_skus, "_tot").drop(columns=["_tot"])

    done = _already_forecasted()
    console.print(f"  {len(df):,} SKUs in M5  |  {len(done):,} already forecasted")

    console.print("  Melting to long format ...")
    df_melt = (
        df[["item_id", "store_id"] + day_cols]
        .melt(id_vars=["item_id", "store_id"], var_name="d", value_name="y")
    )
    df_melt["unique_id"] = df_melt["item_id"] + "_" + df_melt["store_id"]
    df_melt["ds"]        = df_melt["d"].map(d2date)
    df_melt = df_melt.dropna(subset=["ds"]).drop(columns=["item_id", "store_id", "d"])
    df_melt["ds"] = pd.to_datetime(df_melt["ds"])
    df_melt["y"]  = df_melt["y"].astype(float)

    # Update demand_history — skip if already populated
    with get_db() as conn:
        existing_m5 = conn.execute(
            "SELECT COUNT(*) FROM demand_history WHERE store_id='m5_walmart'"
        ).fetchone()[0]

    if existing_m5 < 1_000_000:
        console.print("  Updating demand_history (M5) ...")
        hist_rows = [
            (r.unique_id, r.ds.strftime("%Y-%m-%d"), int(r.y), "m5_walmart")
            for r in df_melt.itertuples()
        ]
        with get_db() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO demand_history (sku,date,units_sold,store_id) VALUES(?,?,?,?)",
                hist_rows,
            )
        console.print(f"  demand_history (M5): {len(hist_rows):,} rows inserted")
    else:
        console.print(f"  demand_history (M5): {existing_m5:,} rows already present — skipping")

    # Filter to series not yet forecasted and long enough
    valid  = (
        df_melt.groupby("unique_id")
               .filter(lambda g: len(g) >= MIN_LEN and g["unique_id"].iloc[0] not in done)
    )
    console.print(f"  {valid['unique_id'].nunique():,} M5 SKUs to forecast on GPU")

    if valid.empty:
        console.print("  [green]All M5 SKUs already forecasted.[/green]")
        return

    pred = _gpu_forecast(valid, HORIZON)
    if pred is None:
        pred = _rolling_fallback(valid, HORIZON)
    stored = _store(pred)
    console.print(f"  [green]M5 forecast rows stored: {stored:,}[/green]")


# ── History (Olist + DataCo + UCI) ───────────────────────────────────────────

def train_from_history(n_skus: int = 0) -> None:
    console.print("[bold]History — Olist + DataCo + UCI Retail[/bold]")
    with get_db() as conn:
        rows = conn.execute(
            "SELECT sku, date, SUM(units_sold) as u "
            "FROM demand_history WHERE store_id != 'm5_walmart' "
            "GROUP BY sku, date ORDER BY sku, date"
        ).fetchall()

    if not rows:
        console.print("  No non-M5 history found")
        return

    df = pd.DataFrame(rows, columns=["unique_id", "ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"]  = df["y"].astype(float)

    done      = _already_forecasted()
    all_skus  = df.groupby("unique_id")["y"].sum().nlargest(n_skus or 999_999).index
    valid     = df[df["unique_id"].isin(all_skus) & ~df["unique_id"].isin(done)]
    valid     = valid.groupby("unique_id").filter(lambda g: len(g) >= MIN_LEN)

    n_todo = valid["unique_id"].nunique()
    console.print(f"  {len(all_skus):,} total  |  {n_todo:,} not yet forecasted")

    if valid.empty:
        console.print("  [green]All history SKUs already forecasted.[/green]")
        return

    pred   = _gpu_forecast(valid, HORIZON) or _rolling_fallback(valid, HORIZON)
    stored = _store(pred)
    console.print(f"  [green]history forecast rows stored: {stored:,}[/green]")


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    m5_only: bool = typer.Option(False, "--m5-only"),
    n_skus:  int  = typer.Option(0,     "--n-skus", help="Limit SKUs per source (0=all)"),
    retrain: bool = typer.Option(False, "--retrain", help="Clear existing forecasts and retrain from scratch on GPU"),
) -> None:
    init_db()

    if retrain:
        with get_db() as conn:
            n = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
            conn.execute("DELETE FROM forecasts")
        console.print(f"[yellow]--retrain: cleared {n:,} existing forecasts[/yellow]")

    if not m5_only:
        train_from_history(n_skus)

    train_m5(n_skus)

    with get_db() as conn:
        n    = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        skus = conn.execute("SELECT COUNT(DISTINCT sku) FROM forecasts").fetchone()[0]
    console.print(f"\n[bold green]Done.[/bold green]  "
                  f"{skus:,} SKUs x {HORIZON} days = {n:,} forecast rows")


if __name__ == "__main__":
    app()
