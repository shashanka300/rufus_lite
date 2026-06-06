#!/usr/bin/env python3
"""
Drop rufus_products Qdrant points that have no image_url.

Keeps only the ~190K products that have a real image URL patched in.
Safe to re-run: if all remaining products have images, it prints "Nothing to delete."

Run:
  uv run python scripts/drop_imageless_products.py
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FilterSelector,
    IsEmptyCondition,
    PayloadField,
)
from rich.console import Console

console = Console(highlight=False, emoji=False)
COLLECTION = "rufus_products"


def main() -> None:
    client = QdrantClient(host="localhost", port=6333, timeout=120)

    total_before = client.count(COLLECTION, exact=True).count
    console.print(f"[bold]rufus_products before:[/bold] {total_before:,} points")

    # --- exact count of points without a real image_url --------------------
    # IsEmpty matches: key missing | value is null | value is "" | value is []
    no_image_filter = Filter(
        must=[IsEmptyCondition(is_empty=PayloadField(key="image_url"))]
    )
    no_image_count = client.count(COLLECTION, count_filter=no_image_filter, exact=True).count
    with_image = total_before - no_image_count

    console.print(f"  With image_url : {with_image:,}  ({with_image/total_before*100:.1f}%)")
    console.print(f"  Without        : {no_image_count:,}  ({no_image_count/total_before*100:.1f}%)")

    if no_image_count == 0:
        console.print("[green]Nothing to delete — all products already have images.[/green]")
        return

    console.print(f"\nDeleting {no_image_count:,} image-less points …")
    client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(filter=no_image_filter),
        wait=True,
    )

    total_after = client.count(COLLECTION, exact=True).count
    console.print(f"[bold green]Done.[/bold green]  rufus_products now has {total_after:,} points.")

    # spot-check: every remaining point should have a real image_url
    result, _ = client.scroll(
        COLLECTION, limit=10, with_payload=["product_id", "image_url"], with_vectors=False
    )
    missing = [pt.payload.get("product_id") for pt in result if not pt.payload.get("image_url")]
    if missing:
        console.print(f"[yellow]WARNING: {len(missing)} spot-check points still have no image_url: {missing}[/yellow]")
    else:
        console.print(f"[green]Spot-check OK — all {len(result)} sampled points have image_url.[/green]")


if __name__ == "__main__":
    main()
