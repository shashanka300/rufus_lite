"""
Session cart — pure in-memory state, persisted via LangGraph MemorySaver.

No backend required: cart lives in ShoppingState.cart and survives
turn-to-turn within a session. Cross-session persistence would require
a real user store (not yet built).
"""

from __future__ import annotations


def add_item(cart: list[dict], product: dict) -> list[dict]:
    """Add product to cart; increment qty if already present."""
    pid = product.get("product_id") or product.get("sku") or product.get("title", "")
    for item in cart:
        if item.get("product_id") == pid or item.get("title") == product.get("title"):
            item["qty"] = item.get("qty", 1) + 1
            return cart
    cart.append({**product, "qty": 1})
    return cart


def remove_item(cart: list[dict], query: str) -> list[dict]:
    """Remove first item whose title contains query (case-insensitive)."""
    q = query.lower()
    return [i for i in cart if q not in (i.get("title") or "").lower()]


def clear_cart(cart: list[dict]) -> list[dict]:
    return []


def cart_total(cart: list[dict]) -> float:
    total = 0.0
    for item in cart:
        price = item.get("unit_price") or item.get("price") or 0
        total += float(price) * item.get("qty", 1)
    return round(total, 2)


def format_cart(cart: list[dict]) -> str:
    if not cart:
        return "Your cart is empty."
    lines = [f"Cart ({len(cart)} item{'s' if len(cart) != 1 else ''}):"]
    for i, item in enumerate(cart, 1):
        title = item.get("title", "Unknown product")[:50]
        qty   = item.get("qty", 1)
        price = item.get("unit_price") or item.get("price")
        line  = f"  {i}. {title} × {qty}"
        if price:
            line += f"  (${float(price):.2f} each)"
        lines.append(line)
    total = cart_total(cart)
    if total > 0:
        lines.append(f"  Total: ${total:.2f}")
    return "\n".join(lines)
