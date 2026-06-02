#!/usr/bin/env python3
"""
Ingest SQID CLIP image embeddings into a separate Qdrant collection (rufus_clip).

The SQID dataset ships pre-computed clip-vit-large-patch14 vectors — no GPU
needed here, we are just copying numpy arrays into Qdrant.

Usage:
  uv run python scripts/ingest_clip.py            # full 164 900 products
  uv run python scripts/ingest_clip.py --reset    # drop + rebuild
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

app = typer.Typer(help="Ingest SQID CLIP embeddings into Qdrant.")
console = Console()

DATA_DIR = Path("data")
SQID_DIR = DATA_DIR / "sqid" / "data"
ESCI_DIR = DATA_DIR / "esci" / "shopping_queries_dataset"
QDRANT_PATH = DATA_DIR / "qdrant_storage"

COLLECTION = "rufus_clip"
VECTOR_DIM = 768   # clip-vit-large-patch14 output dim
BATCH_SIZE = 4096  # pure numpy → can go large


@app.command()
def ingest(
    reset: bool = typer.Option(False, "--reset", help="Drop and rebuild the collection"),
    batch_size: int = typer.Option(BATCH_SIZE, "--batch-size"),
):
    """Upsert SQID CLIP image embeddings into a local Qdrant collection."""

    # ── Load SQID features ─────────────────────────────────────────────────
    console.rule("[bold]Loading SQID features[/bold]")
    feat_path = SQID_DIR / "product_features.parquet"
    url_path = SQID_DIR / "product_image_urls.parquet"

    if not feat_path.exists():
        console.print(f"[red]Not found: {feat_path}[/red]")
        console.print("Run: uv run python scripts/download_datasets.py sqid")
        raise typer.Exit(1)

    sqid = pd.read_parquet(feat_path)
    urls = pd.read_parquet(url_path).set_index("product_id")
    console.print(f"  SQID products: {len(sqid):,}")

    # ── Join with ESCI for product metadata ────────────────────────────────
    console.rule("[bold]Joining with ESCI metadata[/bold]")
    esci = pd.read_parquet(ESCI_DIR / "shopping_queries_dataset_products.parquet")
    esci_us = esci[esci["product_locale"] == "us"].set_index("product_id")

    # only keep rows that exist in ESCI
    sqid = sqid[sqid["product_id"].isin(esci_us.index)].reset_index(drop=True)
    console.print(f"  After ESCI join: {len(sqid):,} products")

    # Drop rows where clip_image_features contains NaN (missing images)
    before = len(sqid)
    sqid = sqid[sqid["clip_image_features"].apply(
        lambda v: v is not None and not any(x != x for x in v)  # x!=x is nan check
    )].reset_index(drop=True)
    console.print(f"  After NaN filter: {len(sqid):,} products ({before - len(sqid):,} dropped)")

    # ── Qdrant setup ───────────────────────────────────────────────────────
    console.rule("[bold]Qdrant setup[/bold]")
    try:
        client = QdrantClient(
            host="localhost", port=6333, grpc_port=6334,
            prefer_grpc=True, timeout=300,
        )
        client.get_collections()
        console.print("  [green]Using Qdrant server (gRPC) at localhost:6334[/green]")
    except Exception:
        console.print("  [yellow]Qdrant server not found — using local file mode[/yellow]")
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(QDRANT_PATH))

    if reset and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        console.print(f"  [yellow]Dropped '{COLLECTION}'[/yellow]")

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        console.print(f"  [green]Created '{COLLECTION}' (dim={VECTOR_DIM}, cosine)[/green]")
    else:
        info = client.get_collection(COLLECTION)
        if not reset:
            console.print(
                f"  [yellow]'{COLLECTION}' already exists "
                f"({info.points_count:,} points) — resuming[/yellow]"
            )
            # skip already-ingested
            seen: set[str] = set()
            offset = None
            while True:
                result, offset = client.scroll(
                    collection_name=COLLECTION, limit=10_000, offset=offset,
                    with_payload=["product_id"], with_vectors=False,
                )
                for pt in result:
                    if pt.payload:
                        seen.add(pt.payload.get("product_id", ""))
                if offset is None:
                    break
            before = len(sqid)
            sqid = sqid[~sqid["product_id"].isin(seen)].reset_index(drop=True)
            console.print(f"  Skipping {before - len(sqid):,} already-ingested")

    if sqid.empty:
        console.print("[green]Nothing new to ingest — collection is up to date.[/green]")
        raise typer.Exit(0)

    console.print(f"  Will ingest [bold]{len(sqid):,}[/bold] products")

    # ── Upsert ─────────────────────────────────────────────────────────────
    console.rule("[bold]Upserting CLIP vectors[/bold]")
    n_batches = (len(sqid) + batch_size - 1) // batch_size
    total_upserted = 0

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(), MofNCompleteColumn(), TaskProgressColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Upserting", total=n_batches)

        for i in range(n_batches):
            batch = sqid.iloc[i * batch_size: (i + 1) * batch_size]

            points = []
            for _, row in batch.iterrows():
                pid = row["product_id"]
                meta = esci_us.loc[pid]
                img_url = urls.loc[pid, "image_url"] if pid in urls.index else None

                # Use image embedding as the primary vector (visual-semantic search)
                vec = np.asarray(row["clip_image_features"], dtype=np.float32).tolist()

                points.append(PointStruct(
                    id=int(hashlib.sha256(pid.encode()).hexdigest()[:16], 16),
                    vector=vec,
                    payload={
                        "product_id": pid,
                        "product_title": meta["product_title"],
                        "product_brand": meta["product_brand"] if pd.notna(meta["product_brand"]) else None,
                        "product_color": meta["product_color"] if pd.notna(meta["product_color"]) else None,
                        "product_locale": "us",
                        "product_bullet_point": (
                            str(meta["product_bullet_point"])[:1000]
                            if pd.notna(meta["product_bullet_point"]) else None
                        ),
                        "image_url": img_url,
                    },
                ))

            client.upsert(collection_name=COLLECTION, points=points)
            total_upserted += len(points)
            progress.advance(task)

    info = client.get_collection(COLLECTION)
    console.print()
    console.print(f"[green bold]Done![/green bold] Upserted {total_upserted:,} CLIP vectors.")
    console.print(f"Collection '{COLLECTION}' now has [bold]{info.points_count:,}[/bold] points.")


if __name__ == "__main__":
    app()
