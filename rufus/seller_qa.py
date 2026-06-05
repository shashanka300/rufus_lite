"""
Seller Q&A mock — category-level question/answer templates.

Simulates the Amazon community Q&A / seller-submitted FAQ feature.
Real data would come from the product detail pages; this module
provides plausible answers by category so the feature works immediately.
"""

from __future__ import annotations

import re

_QA_TEMPLATES: dict[str, list[dict]] = {
    "electronics": [
        {"q": "warranty",      "a": "Most electronics include a 1-year manufacturer warranty. Extended warranty plans are available at checkout."},
        {"q": "compatible",    "a": "Check the product listing for a full compatibility list. Most devices support iOS 14+ and Android 8+."},
        {"q": "battery",       "a": "Battery life varies by usage. See the listed specs for rated hours; real-world life is typically 80–90% of that."},
        {"q": "return",        "a": "Electronics can be returned within 30 days of delivery if unopened or defective."},
        {"q": "international", "a": "This product uses a standard voltage adapter. International use may require a plug adapter."},
    ],
    "headphones": [
        {"q": "noise cancel",  "a": "Active noise cancellation (ANC) is listed in the product specs. Passive isolation depends on fit."},
        {"q": "microphone",    "a": "Built-in microphone quality varies. Check the specs for mic sensitivity and frequency response."},
        {"q": "connect",       "a": "Bluetooth headphones typically connect to 2 devices simultaneously. Check the product page for multipoint support."},
    ],
    "clothing": [
        {"q": "size",          "a": "Refer to the size chart in the product images. When between sizes, most buyers size up."},
        {"q": "material",      "a": "Material composition is listed under product details. Wash instructions are printed on the tag."},
        {"q": "return",        "a": "Unworn clothing with original tags can be returned within 30 days."},
        {"q": "shrink",        "a": "To prevent shrinkage, wash in cold water and air dry. Avoid high heat."},
    ],
    "home_kitchen": [
        {"q": "dishwasher",    "a": "Check the product listing for dishwasher-safe status. Hand wash is recommended for items with non-stick coatings."},
        {"q": "dimensions",    "a": "Exact dimensions are listed in the product details section."},
        {"q": "warranty",      "a": "Kitchen appliances typically include a 1-year limited warranty against manufacturing defects."},
    ],
    "sports": [
        {"q": "size",          "a": "Use the manufacturer's size guide in the product images. Athletic fit items often run small."},
        {"q": "waterproof",    "a": "Water resistance rating (IPX) is listed in the specs. IPX4 = splash-proof; IPX7 = submersible to 1m."},
        {"q": "weight",        "a": "Product weight is listed in the specifications. Shipping weight includes packaging."},
    ],
    "default": [
        {"q": "return",        "a": "Most items can be returned within 30 days of delivery. See the return policy for item-specific exceptions."},
        {"q": "warranty",      "a": "Manufacturer warranty details are listed on the product page. Contact the seller for extended options."},
        {"q": "availability",  "a": "Availability and estimated delivery dates are shown at checkout based on your location."},
        {"q": "gift",          "a": "Gift wrapping and a personalised message can be added at checkout."},
    ],
}


def _category_key(category: str) -> str:
    c = (category or "").lower()
    for key in _QA_TEMPLATES:
        if key != "default" and key in c:
            return key
    return "default"


def answer_question(question: str, category: str = "") -> str | None:
    """
    Return a template answer if the question matches a known pattern.
    Returns None if no template matches (caller should fall back to LLM).
    """
    q = question.lower()
    key = _category_key(category)
    templates = _QA_TEMPLATES.get(key, []) + _QA_TEMPLATES["default"]

    for tmpl in templates:
        if tmpl["q"] in q:
            suffix = " *(Community Q&A — verify with seller for specifics.)*"
            return tmpl["a"] + suffix

    return None


def get_category_faqs(category: str, n: int = 3) -> list[dict]:
    """Return top N FAQ pairs for a category — useful for proactive suggestions."""
    key = _category_key(category)
    templates = _QA_TEMPLATES.get(key, _QA_TEMPLATES["default"])
    return [{"question": t["q"].title(), "answer": t["a"]} for t in templates[:n]]
