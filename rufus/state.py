"""
LangGraph state schema for the Rufus conversational shopping assistant.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ShoppingState(TypedDict):
    # append-only
    messages:    Annotated[list[dict], operator.add]
    # overwritten each turn
    intent:      str   # search|followup|qa|compare|chitchat|gift_search
                       # add_to_cart|view_cart
                       # check_stock|reorder_alert|demand_forecast|supplier_query|sc_analytics
    query:       str
    products:    list  # list[Product]; persists on followup
    filters:     dict  # brand, color, price_max, category
    # personalization
    session_id:  str   # set by server; drives preference profile
    # cart
    cart:        list  # list[dict] — persists across turns via MemorySaver
    # supply chain
    sc_items:    list
    sc_context:  str
