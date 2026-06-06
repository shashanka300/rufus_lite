#!/usr/bin/env python3
"""
Rebuild rufus_products from rufus_clip.

rufus_clip has 156,542 ESCI products that all have verified image URLs.
This script scrolls rufus_clip for their metadata, re-embeds the product text
with BGE-M3 (fp16, RTX 5090), and upserts into a freshly rebuilt rufus_products.

After this runs:
  - rufus_products has 156 K products, every one with an image_url
  - rufus_clip is unchanged (CLIP vectors for visual search)
  - RRF fusion, reranker, RAG all continue to work as before

Run:
  uv run python scripts/rebuild_products_from_clip.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
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

import rufus.hardware  # TF32 + cuDNN benchmark

console = Console(highlight=False, emoji=False)

SRC_COLLECTION = "rufus_clip"
DST_COLLECTION = "rufus_products"
EMBED_MODEL    = "BAAI/bge-m3"
EMBED_DIM      = 1024
SCROLL_BATCH   = 2000   # points per Qdrant scroll call
EMBED_BATCH    = 128    # texts per BGE-M3 call  (RTX 5090 fp16)
UPSERT_BATCH   = 256    # points per Qdrant upsert call


def _pid_to_int(pid: str) -> int:
    """Stable integer ID: first 16 hex chars of SHA-256."""
    return int(hashlib.sha256(pid.encode()).hexdigest()[:16], 16)


def _build_text(payload: dict) -> str:
    title  = payload.get("product_title") or ""
    bullet = payload.get("product_bullet_point") or ""
    desc   = payload.get("product_description") or ""
    return f"{title}. {bullet} {desc}".strip()[:512]


def main() -> None:
    # ── Qdrant connection ──────────────────────────────────────────────────
    client = QdrantClient(host="localhost", port=6333, timeout=300)
    src_info = client.get_collection(SRC_COLLECTION)
    src_total = src_info.points_count
    console.print(f"[bold]{SRC_COLLECTION}[/bold]: {src_total:,} points (source)")

    # ── Rebuild rufus_products ─────────────────────────────────────────────
    if client.collection_exists(DST_COLLECTION):
        client.delete_collection(DST_COLLECTION)
        console.print(f"[yellow]Dropped existing '{DST_COLLECTION}'[/yellow]")

    client.create_collection(
        collection_name=DST_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    console.print(f"[green]Created '{DST_COLLECTION}' (dim={EMBED_DIM}, cosine)[/green]")

    # ── Load BGE-M3 ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Loading BGE-M3 on {device} (fp16)…")
    model = SentenceTransformer(
        EMBED_MODEL, device=device,
        model_kwargs={"torch_dtype": torch.float16},
    )
    console.print(f"[green]BGE-M3 loaded[/green]")

    # ── Scroll + embed + upsert ────────────────────────────────────────────
    offset = None
    buf_payloads: list[dict] = []
    buf_pids:     list[str]  = []
    total_upserted = 0

    console.print(f"\nScrolling {SRC_COLLECTION}, embedding with BGE-M3, upserting …")

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(), MofNCompleteColumn(), TaskProgressColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Products", total=src_total)

        def _flush():
            nonlocal total_upserted
            if not buf_pids:
                return
            texts = [_build_text(p) for p in buf_payloads]
            vecs  = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=EMBED_BATCH,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            points = [
                PointStruct(
                    id=_pid_to_int(pid),
                    vector=vec.tolist(),
                    payload={**payload},  # already contains image_url
                )
                for pid, payload, vec in zip(buf_pids, buf_payloads, vecs)
            ]
            # upsert in sub-batches to stay under HTTP limits
            for i in range(0, len(points), UPSERT_BATCH):
                client.upsert(DST_COLLECTION, points=points[i:i+UPSERT_BATCH])
            total_upserted += len(points)
            progress.advance(task, len(points))
            buf_pids.clear()
            buf_payloads.clear()

        while True:
            result, offset = client.scroll(
                SRC_COLLECTION,
                limit=SCROLL_BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in result:
                if not pt.payload:
                    continue
                pid = pt.payload.get("product_id", "")
                if not pid:
                    continue
                buf_pids.append(pid)
                buf_payloads.append(pt.payload)
                if len(buf_pids) >= SCROLL_BATCH:
                    _flush()

            if offset is None:
                break

        _flush()  # leftover

    final = client.count(DST_COLLECTION, exact=True).count
    console.print(f"\n[bold green]Done.[/bold green]  "
                  f"'{DST_COLLECTION}' now has {final:,} points (all with image_url).")

    # spot-check
    result, _ = client.scroll(DST_COLLECTION, limit=5,
                               with_payload=["product_id", "image_url"], with_vectors=False)
    console.print("\nSpot-check:")
    for pt in result:
        pid = pt.payload.get("product_id", "?")
        img = pt.payload.get("image_url") or "MISSING"
        console.print(f"  {pid}: {img[:70]}")


if __name__ == "__main__":
    main()
