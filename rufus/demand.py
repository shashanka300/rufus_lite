"""
Demand forecasting for supply chain queries.

Two tiers:
  1. Stored Prophet forecasts (from scripts/train_forecasts.py) — used when available
  2. Rolling average fallback — always available, reasonable for most queries

Public API
----------
  forecast_sku(sku, days=30)   -> ForecastResult
  bulk_reorder_check(skus)     -> list[ReorderAlert]
  category_demand_summary(cat) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from rufus.inventory import (
    avg_daily_demand,
    get_db,
    get_forecast,
    get_item,
    get_low_stock,
    reorder_recommendation,
    summary_stats,
)


@dataclass
class ForecastResult:
    sku: str
    title: str
    horizon_days: int
    total_predicted: float
    daily_avg: float
    days_of_stock: float
    reorder_urgency: str   # "critical" | "soon" | "ok"
    source: str

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ReorderAlert:
    sku: str
    title: str
    category: str
    qty_on_hand: int
    reorder_pt: int
    days_of_stock: float
    recommended_qty: int
    urgency: str

    def to_dict(self) -> dict:
        return self.__dict__


def forecast_sku(sku: str, horizon_days: int = 30) -> ForecastResult | None:
    item = get_item(sku)
    if not item:
        return None

    fc = get_forecast(sku, horizon_days)
    daily = fc["daily_avg"] or 0.1
    dos = round(item["qty_on_hand"] / daily, 1)
    lead = item.get("lead_time") or 14

    if dos <= lead:
        urgency = "critical"
    elif dos <= lead * 2:
        urgency = "soon"
    else:
        urgency = "ok"

    return ForecastResult(
        sku=sku,
        title=item.get("title", sku),
        horizon_days=horizon_days,
        total_predicted=fc["total_predicted"],
        daily_avg=fc["daily_avg"],
        days_of_stock=dos,
        reorder_urgency=urgency,
        source=fc["source"],
    )


def bulk_reorder_check(skus: list[str] | None = None) -> list[ReorderAlert]:
    """
    Check reorder status for a list of SKUs (or all low-stock items if None).
    Returns alerts sorted by urgency.
    """
    if skus is None:
        items = get_low_stock(limit=50)
        skus = [i["sku"] for i in items]

    alerts: list[ReorderAlert] = []
    for sku in skus:
        rec = reorder_recommendation(sku)
        if not rec:
            continue
        daily = rec["avg_daily_demand"] or 0.1
        lead = rec["lead_time_days"]
        dos = rec["days_of_stock"]

        urgency = "critical" if dos <= lead else ("soon" if dos <= lead * 2 else "ok")
        alerts.append(ReorderAlert(
            sku=sku,
            title=rec["title"],
            category=rec.get("category", ""),
            qty_on_hand=rec["qty_on_hand"],
            reorder_pt=rec["reorder_pt"],
            days_of_stock=dos,
            recommended_qty=rec["recommended_order_qty"],
            urgency=urgency,
        ))

    alerts.sort(key=lambda a: (a.urgency != "critical", a.urgency != "soon", a.days_of_stock))
    return alerts


def category_demand_summary(category: str, top_n: int = 5) -> dict:
    """Aggregate demand stats for a product category."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT i.sku, i.title, i.qty_on_hand, i.reorder_pt,
                      COALESCE(SUM(dh.units_sold), 0) as total_sold_30d
               FROM inventory i
               LEFT JOIN demand_history dh
                 ON dh.sku = i.sku
                 AND dh.date >= date('now', '-30 days')
               WHERE lower(i.category) LIKE lower(?)
               GROUP BY i.sku
               ORDER BY total_sold_30d DESC
               LIMIT ?""",
            (f"%{category}%", top_n),
        ).fetchall()

    items = [dict(r) for r in rows]
    total_units = sum(r["qty_on_hand"] for r in items)
    total_sold = sum(r["total_sold_30d"] for r in items)
    low_stock_count = sum(1 for r in items if r["qty_on_hand"] < r["reorder_pt"])

    return {
        "category": category,
        "top_items": items,
        "total_units_on_hand": total_units,
        "total_sold_last_30d": total_sold,
        "low_stock_count": low_stock_count,
    }


def inventory_health_report() -> dict:
    """High-level KPIs for the inventory dashboard."""
    stats = summary_stats()
    critical = bulk_reorder_check()
    critical_count = sum(1 for a in critical if a.urgency == "critical")
    soon_count = sum(1 for a in critical if a.urgency == "soon")
    return {
        **stats,
        "critical_reorder": critical_count,
        "reorder_soon": soon_count,
        "top_critical": [a.to_dict() for a in critical[:5] if a.urgency == "critical"],
    }
