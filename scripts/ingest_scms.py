#!/usr/bin/env python3
"""
Ingest SCMS Delivery History into rufus_sc.db suppliers table.

SCMS = Supply Chain Management System (USAID PEPFAR).
Provides real vendor lead times, freight modes, and unit pricing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console

from rufus.inventory import get_db, init_db

console = Console(highlight=False, emoji=False)

# Try both possible locations
SCMS_PATHS = [
    Path("data/supply_chain/scms/scms_delivery_history.csv"),
    Path("data/supply_chain/scms/SCMS_Delivery_History_Dataset.csv"),
    *Path("data/supply_chain/scms").glob("*.csv"),
]


def _find_scms() -> Path | None:
    for p in SCMS_PATHS:
        if p.exists():
            return p
    return None


def main() -> None:
    console.print("[bold]Ingest: SCMS -> rufus_sc.db suppliers[/bold]")
    init_db()

    path = _find_scms()
    if not path:
        console.print("[red]SCMS CSV not found — skipping[/red]")
        return

    console.print(f"Loading {path.name} ...")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    console.print(f"  {len(df):,} rows, columns: {df.columns.tolist()[:8]} ...")

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]

    # Key columns (SCMS uses varied naming across versions)
    vendor_col   = next((c for c in df.columns if "vendor" in c), None)
    country_col  = next((c for c in df.columns if "country" in c), None)
    freight_col  = next((c for c in df.columns if "shipment_mode" in c or "freight" in c), None)
    sched_col    = next((c for c in df.columns if "scheduled" in c and "date" in c), None)
    deliv_col    = next((c for c in df.columns if "delivered" in c and "date" in c), None)
    price_col    = next((c for c in df.columns if "unit_price" in c or "pack_price" in c), None)

    if not vendor_col:
        console.print("[red]Could not identify vendor column — skipping[/red]")
        return

    # Compute lead time as delivered - scheduled (days)
    if sched_col and deliv_col:
        df[sched_col] = pd.to_datetime(df[sched_col], errors="coerce", dayfirst=False, format="mixed")
        df[deliv_col] = pd.to_datetime(df[deliv_col], errors="coerce", dayfirst=False, format="mixed")
        df["lead_days"] = (df[deliv_col] - df[sched_col]).dt.days

    # Aggregate per vendor
    agg: dict[str, dict] = {}
    for row in df.itertuples():
        vendor = str(getattr(row, vendor_col, "")).strip()
        if not vendor or vendor == "nan":
            continue
        if vendor not in agg:
            agg[vendor] = {"lead_times": [], "prices": [], "country": "", "freight": ""}
        if sched_col and deliv_col and hasattr(row, "lead_days"):
            ld = getattr(row, "lead_days", None)
            if ld and 0 < ld < 365:
                agg[vendor]["lead_times"].append(ld)
        if country_col:
            agg[vendor]["country"] = str(getattr(row, country_col, "")).strip() or agg[vendor]["country"]
        if freight_col:
            agg[vendor]["freight"] = str(getattr(row, freight_col, "")).strip() or agg[vendor]["freight"]
        if price_col:
            try:
                p = float(str(getattr(row, price_col, "0")).replace(",", "").replace("$", ""))
                if p > 0:
                    agg[vendor]["prices"].append(p)
            except (ValueError, TypeError):
                pass

    supplier_rows = []
    for vendor, data in agg.items():
        avg_lead = int(sum(data["lead_times"]) / len(data["lead_times"])) if data["lead_times"] else 21
        avg_price = round(sum(data["prices"]) / len(data["prices"]), 2) if data["prices"] else 0.0
        import hashlib
        sid = hashlib.sha256(vendor.encode()).hexdigest()[:16]
        supplier_rows.append((
            sid,
            vendor[:80],
            data["country"][:60] or "Unknown",
            avg_lead,
            data["freight"][:40] or "Air",
            avg_price,
        ))

    with get_db() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO suppliers
               (supplier_id, name, country, lead_time_days, freight_mode, avg_unit_price)
               VALUES (?,?,?,?,?,?)""",
            supplier_rows,
        )
    console.print(f"  [green]inserted[/green] {len(supplier_rows):,} SCMS supplier rows")

    with get_db() as conn:
        n = conn.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
    console.print(f"  suppliers total: {n:,}")
    console.print("[bold green]SCMS ingest complete.[/bold green]")


if __name__ == "__main__":
    main()
