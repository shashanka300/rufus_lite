"""
Supply chain context formatter for LLM generation.
Mirrors rufus/rag.py but for inventory/demand/supplier data.
"""

from __future__ import annotations

SC_SYSTEM_PROMPT = """\
You are Rufus, an AI assistant for supply chain and inventory operations.

RULES — follow every time:
1. Report ONLY data that is explicitly provided. Never invent SKU names, supplier names, \
quantities, or any other values not present in the data.
2. If a SKU or supplier name looks like a hash/code (e.g. "9dc1a7de"), report it verbatim — \
do NOT substitute a made-up human-readable name.
3. If the data section says "No inventory data found" or "No supplier data found", say so \
clearly — do NOT invent placeholder data.
4. Be precise with numbers. Never round or estimate if exact data is given.
5. Flag CRITICAL items (out of stock, days_of_stock < lead_time) at the top.
6. Keep responses under 150 words.
7. Do not start with "I" or "Sure" or filler phrases.
8. Do not repeat the user's question.

OUTPUT FORMAT by intent:
- check_stock: 1-2 sentences: status (in stock / low / out), exact qty, days of stock
- reorder_alert: bullet list — SKU/name as given, qty on hand, days of stock
- demand_forecast: 2-3 sentences: predicted units, daily avg, reorder urgency
- supplier_query: bullet list — supplier name as given, country, lead time, freight mode
- sc_analytics: summary stats first, then top items needing attention\
"""


def format_inventory_context(items: list[dict]) -> str:
    if not items:
        return "No inventory data found."
    lines = []
    for item in items:
        dos = item.get("days_of_stock") or "?"
        status = "OUT OF STOCK" if item.get("qty_on_hand", 0) <= 0 else (
            "LOW STOCK" if item.get("qty_on_hand", 0) < item.get("reorder_pt", 0) else "OK"
        )
        line = (
            f"SKU: {item['sku'][:16]}  |  {item.get('title','?')[:50]}\n"
            f"  Category: {item.get('category','?')}  |  Status: {status}\n"
            f"  On hand: {item.get('qty_on_hand','?')} units"
            f"  |  Reorder at: {item.get('reorder_pt','?')}"
            f"  |  Days of stock: {dos}"
        )
        price = item.get("unit_price")
        if price:
            line += f"  |  Price: ${price:.2f}"
        lines.append(line)
    return "\n\n".join(lines)


def format_reorder_context(alerts: list) -> str:
    if not alerts:
        return "No reorder alerts at this time."
    lines = ["Reorder Alerts:"]
    for a in alerts:
        d = a.to_dict() if hasattr(a, "to_dict") else a
        urgency_tag = "[CRITICAL]" if d.get("urgency") == "critical" else (
            "[SOON]" if d.get("urgency") == "soon" else "[OK]"
        )
        lines.append(
            f"{urgency_tag} {d.get('title','?')[:50]}\n"
            f"  On hand: {d.get('qty_on_hand','?')}  |  "
            f"Reorder pt: {d.get('reorder_pt','?')}  |  "
            f"Days of stock: {d.get('days_of_stock','?')}  |  "
            f"Suggested order: {d.get('recommended_qty','?')} units"
        )
    return "\n".join(lines)


def format_forecast_context(fc) -> str:
    if fc is None:
        return "No forecast data available."
    d = fc.to_dict() if hasattr(fc, "to_dict") else fc
    return (
        f"Demand Forecast — {d.get('title','?')[:50]}\n"
        f"  SKU: {d.get('sku','?')[:16]}\n"
        f"  Forecast horizon: {d.get('horizon_days','?')} days\n"
        f"  Predicted demand: {d.get('total_predicted','?')} units total "
        f"({d.get('daily_avg','?')} units/day avg)\n"
        f"  Current days of stock: {d.get('days_of_stock','?')}\n"
        f"  Reorder urgency: {d.get('reorder_urgency','?').upper()}\n"
        f"  Forecast method: {d.get('source','?')}"
    )


def format_supplier_context(suppliers: list[dict]) -> str:
    if not suppliers:
        return "No supplier data found."
    lines = ["Available Suppliers:"]
    for s in suppliers:
        lines.append(
            f"  {s.get('name','?')[:40]}  |  Country: {s.get('country','?')}"
            f"  |  Lead time: {s.get('lead_time_days','?')} days"
            f"  |  Freight: {s.get('freight_mode','?')}"
            f"  |  Avg price: ${s.get('avg_unit_price',0):.2f}"
        )
    return "\n".join(lines)


def format_sc_context(
    intent: str,
    inventory: list[dict] | None = None,
    alerts: list | None = None,
    forecast=None,
    suppliers: list[dict] | None = None,
) -> str:
    parts: list[str] = []
    if inventory:
        parts.append(format_inventory_context(inventory))
    if alerts:
        parts.append(format_reorder_context(alerts))
    if forecast:
        parts.append(format_forecast_context(forecast))
    if suppliers:
        parts.append(format_supplier_context(suppliers))
    return "\n\n---\n\n".join(parts) if parts else "No supply chain data available."
