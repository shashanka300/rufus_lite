#!/usr/bin/env python3
"""
Patch image_url into rufus_products Qdrant payloads.

Reads rufus_images.db and calls set_payload() in batches so every
point that has a matching image URL gets it stored directly in Qdrant.
After this runs, _products_to_json() will always find the correct image
via p.image_url without falling back to the DB lookup.

Requires the Qdrant server to be running:
  .\\scripts\\start_qdrant.ps1

Run:  uv run python scripts/update_qdrant_images.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, SetPayload
from rich.console import Console
from rich.progress import track

console = Console(highlight=False, emoji=False)

IMAGES_DB  = Path("data/rufus_images.db")
COLLECTION = "rufus_products"
BATCH_SIZE = 512


def main() -> None:
    console.print("[bold]Patching image_url into Qdrant rufus_products payloads[/bold]")

    if not IMAGES_DB.exists():
        console.print(f"[red]{IMAGES_DB} not found — run scripts/build_image_lookup.py first[/red]")
        return

    # Connect to Qdrant server
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=120)
        client.get_collections()
        console.print("  Qdrant server connected")
    except Exception as e:
        console.print(f"[red]Qdrant server not reachable: {e}[/red]")
        console.print("  Start it with: .\\scripts\\start_qdrant.ps1")
        return

    # Load all product_id -> image_url from images DB
    console.print(f"  Loading {IMAGES_DB} ...")
    conn = sqlite3.connect(str(IMAGES_DB))
    rows = conn.execute("SELECT product_id, image_url FROM images").fetchall()
    conn.close()
    img_map = {pid: url for pid, url in rows}
    console.print(f"  {len(img_map):,} image URLs loaded")

    # Scroll through Qdrant to find points without image_url
    console.print(f"  Scanning {COLLECTION} for points without image_url ...")
    to_update: list[tuple[int, str]] = []  # (qdrant_id, image_url)
    offset = None

    while True:
        result, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=None,
            limit=2000,
            offset=offset,
            with_payload=["product_id", "image_url"],
            with_vectors=False,
        )
        for pt in result:
            if pt.payload.get("image_url"):
                continue  # already has one
            pid = pt.payload.get("product_id", "")
            url = img_map.get(pid)
            if url:
                to_update.append((pt.id, url))
        if offset is None:
            break

    console.print(f"  {len(to_update):,} points to update")

    if not to_update:
        console.print("[green]Nothing to patch.[/green]")
        return

    # Batch update
    updated = 0
    for i in track(range(0, len(to_update), BATCH_SIZE), description="  Patching..."):
        batch = to_update[i : i + BATCH_SIZE]
        ids = [b[0] for b in batch]

        # Group points by same image_url to minimise API calls
        by_url: dict[str, list[int]] = {}
        for pt_id, url in batch:
            by_url.setdefault(url, []).append(pt_id)

        for url, pt_ids in by_url.items():
            client.set_payload(
                collection_name=COLLECTION,
                payload={"image_url": url},
                points=pt_ids,
            )
        updated += len(batch)

    console.print(f"[bold green]Done.[/bold green]  Patched {updated:,} points with image URLs.")


if __name__ == "__main__":
    main()
