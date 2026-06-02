"""
LangGraph state schema for the Rufus conversational shopping assistant.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ShoppingState(TypedDict):
    # append-only: each node adds its new messages, history accumulates
    messages: Annotated[list[dict], operator.add]
    # overwritten each turn
    intent: str        # search | followup | qa | compare | chitchat
    query: str         # cleaned search query from intent classifier
    products: list     # list[Product] from last retrieval; persists on followup
    filters: dict      # extracted preferences: brand, color, price_max, category
