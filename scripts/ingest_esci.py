#!/usr/bin/env python3
"""
Ingest ESCI products into Qdrant using BGE-M3 dense embeddings.

Week 1 pipeline:
  ESCI parquet → text construction → BGE-M3 embed → Qdrant upsert

Usage:
  uv run python scripts/ingest_esci.py            # full English subset
  uv run python scripts/ingest_esci.py --limit 5000  # quick smoke-test
  uv run python scripts/ingest_esci.py --reset    # drop + rebuild collection
"""

from pathlib import Path

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
from sentence_transformers import SentenceTransformer

app = typer.Typer(help="Ingest ESCI products into Qdrant.")
console = Console()

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
ESCI_DIR = DATA_DIR / "esci" / "shopping_queries_dataset"
QDRANT_PATH = DATA_DIR / "qdrant_storage"

# ── Config ─────────────────────────────────────────────────────────────────
COLLECTION = "rufus_products"
EMBED_MODEL = "BAAI/bge-m3"
VECTOR_DIM = 1024          # BGE-M3 dense output dimension
BATCH_SIZE = 512           # large batch for GPU throughput (lower to 128 if OOM)
MAX_BULLETS_CHARS = 600    # truncate long bullet lists


# ── Text construction ──────────────────────────────────────────────────────

def build_product_text(row: pd.Series) -> str:
    """Concatenate the fields BGE-M3 will encode into a single string."""
    parts: list[str] = [row["product_title"]]

    if pd.notna(row["product_brand"]) and row["product_brand"]:
        parts.append(f"Brand: {row['product_brand']}")

    if pd.notna(row["product_color"]) and row["product_color"]:
        parts.append(f"Color: {row['product_color']}")

    if pd.notna(row["product_bullet_point"]) and row["product_bullet_point"]:
        bullets = str(row["product_bullet_point"])[:MAX_BULLETS_CHARS]
        parts.append(bullets)

    return " | ".join(parts)


# ── Qdrant helpers ─────────────────────────────────────────────────────────

def get_client() -> QdrantClient:
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def ensure_collection(client: QdrantClient, reset: bool = False) -> None:
    if reset and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        console.print(f"  [yellow]Dropped existing collection '{COLLECTION}'[/yellow]")

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        console.print(f"  [green]Created collection '{COLLECTION}' (dim={VECTOR_DIM}, cosine)[/green]")
    else:
        info = client.get_collection(COLLECTION)
        console.print(
            f"  [yellow]Collection '{COLLECTION}' already exists "
            f"({info.points_count:,} points) — resuming[/yellow]"
        )


def already_ingested_ids(client: QdrantClient) -> set[str]:
    """Return the set of product_ids already in Qdrant (for resume support)."""
    if not client.collection_exists(COLLECTION):
        return set()
    # scroll through all points to collect stored product_ids
    ids: set[str] = set()
    offset = None
    while True:
        result, offset = client.scroll(
            collection_name=COLLECTION,
            limit=10_000,
            offset=offset,
            with_payload=["product_id"],
            with_vectors=False,
        )
        for pt in result:
            if pt.payload and "product_id" in pt.payload:
                ids.add(pt.payload["product_id"])
        if offset is None:
            break
    return ids


# ── Main ingestion ─────────────────────────────────────────────────────────

@app.command()
def ingest(
    limit: int = typer.Option(0, "--limit", help="Cap number of products (0 = all)"),
    reset: bool = typer.Option(False, "--reset", help="Drop and rebuild the Qdrant collection"),
    locale: str = typer.Option("us", "--locale", help="Product locale to ingest (us/es/jp)"),
    batch_size: int = typer.Option(BATCH_SIZE, "--batch-size"),
):
    """Embed ESCI products with BGE-M3 and upsert into local Qdrant."""

    # ── Load products ──────────────────────────────────────────────────────
    console.rule("[bold]Loading ESCI products[/bold]")
    products_path = ESCI_DIR / "shopping_queries_dataset_products.parquet"
    if not products_path.exists():
        console.print(f"[red]Not found: {products_path}[/red]")
        console.print("Run: uv run python scripts/download_datasets.py esci")
        raise typer.Exit(1)

    df = pd.read_parquet(products_path)
    console.print(f"  Loaded {len(df):,} products total")

    # Filter to requested locale
    df = df[df["product_locale"] == locale].reset_index(drop=True)
    console.print(f"  After locale filter ({locale}): {len(df):,} products")

    if limit:
        df = df.head(limit)
        console.print(f"  Capped to {limit:,} for testing")

    # ── Qdrant setup ───────────────────────────────────────────────────────
    console.rule("[bold]Qdrant setup[/bold]")
    client = get_client()
    ensure_collection(client, reset=reset)

    if not reset:
        seen = already_ingested_ids(client)
        if seen:
            before = len(df)
            df = df[~df["product_id"].isin(seen)].reset_index(drop=True)
            console.print(f"  Skipping {before - len(df):,} already-ingested products")

    if df.empty:
        console.print("[green]Nothing new to ingest — collection is up to date.[/green]")
        raise typer.Exit(0)

    console.print(f"  Will ingest [bold]{len(df):,}[/bold] products")

    # ── Load embedding model ───────────────────────────────────────────────
    console.rule("[bold]Loading BGE-M3[/bold]")
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"  Model: {EMBED_MODEL}  |  Device: [bold]{device}[/bold]")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    console.print("  [green]Model loaded[/green]")

    # ── Build texts ────────────────────────────────────────────────────────
    console.rule("[bold]Building product texts[/bold]")
    texts = df.apply(build_product_text, axis=1).tolist()
    console.print(f"  Built {len(texts):,} text representations")
    console.print(f"  Sample: {texts[0][:120]!r}")

    # ── Embed + upsert in batches ──────────────────────────────────────────
    console.rule("[bold]Embedding + upserting[/bold]")

    n_batches = (len(texts) + batch_size - 1) // batch_size
    total_upserted = 0

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting", total=n_batches)

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(texts))

            batch_texts = texts[start:end]
            batch_rows = df.iloc[start:end]

            # BGE-M3 recommends prepending "Represent this sentence:" for docs
            embeddings = model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            points = []
            for j, (_, row) in enumerate(batch_rows.iterrows()):
                points.append(
                    PointStruct(
                        id=abs(hash(row["product_id"])) % (2**63),
                        vector=embeddings[j].tolist(),
                        payload={
                            "product_id": row["product_id"],
                            "product_title": row["product_title"],
                            "product_brand": row["product_brand"] if pd.notna(row["product_brand"]) else None,
                            "product_color": row["product_color"] if pd.notna(row["product_color"]) else None,
                            "product_locale": row["product_locale"],
                            "product_bullet_point": (
                                str(row["product_bullet_point"])[:1000]
                                if pd.notna(row["product_bullet_point"]) else None
                            ),
                            "product_description": (
                                str(row["product_description"])[:500]
                                if pd.notna(row["product_description"]) else None
                            ),
                        },
                    )
                )

            client.upsert(collection_name=COLLECTION, points=points)
            total_upserted += len(points)
            progress.advance(task)

    # ── Summary ────────────────────────────────────────────────────────────
    info = client.get_collection(COLLECTION)
    console.print()
    console.print(f"[green bold]Done![/green bold] Upserted {total_upserted:,} products.")
    console.print(f"Collection '{COLLECTION}' now has [bold]{info.points_count:,}[/bold] points.")
    console.print(f"Qdrant storage: {QDRANT_PATH}")


if __name__ == "__main__":
    app()
