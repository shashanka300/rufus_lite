#!/usr/bin/env python3
"""
ESCI NDCG@K benchmark for the Rufus retrieval system.

Evaluates BGE-M3 text retrieval vs BGE-M3 + CLIP image fusion on
the ESCI test set (US locale), reporting NDCG@10 for each system.

Usage:
  uv run python scripts/eval_ndcg.py                    # 300 queries, NDCG@10
  uv run python scripts/eval_ndcg.py --n-queries 1000   # more queries
  uv run python scripts/eval_ndcg.py --k 5              # NDCG@5
  uv run python scripts/eval_ndcg.py --seed 99          # different random sample
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import typer
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
from rich.table import Table

from rufus.clip_retriever import CLIPRetriever
from rufus.eval import ESCI_GAINS, EvalResult, ndcg_at_k
from rufus.fusion import rrf_fuse
from rufus.reranker import ProductReranker
from rufus.retriever import ProductRetriever

app = typer.Typer(help="Evaluate Rufus retrieval with NDCG@K.")
console = Console()

DATA_DIR = Path("data")
ESCI_DIR = DATA_DIR / "esci" / "shopping_queries_dataset"
QDRANT_PATH = DATA_DIR / "qdrant_storage"


def _build_relevance_index(examples: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return {query_id: {product_id: gain}} for all queries in examples."""
    index: dict[str, dict[str, float]] = {}
    for _, row in examples.iterrows():
        qid = str(row["query_id"])
        if qid not in index:
            index[qid] = {}
        index[qid][row["product_id"]] = ESCI_GAINS.get(row["esci_label"], 0.0)
    return index


@app.command()
def evaluate(
    n_queries: int = typer.Option(300, "--n-queries", "-n", help="Number of queries to evaluate"),
    k: int = typer.Option(10, "--k", help="Cutoff for NDCG@K"),
    seed: int = typer.Option(42, "--seed", help="Random seed for query sampling"),
    top_k_retrieve: int = typer.Option(20, "--top-k-retrieve", help="Candidates per retriever before fusion"),
    split: str = typer.Option("test", "--split", help="ESCI split to use (test/train)"),
):
    """Benchmark BGE-M3 and BGE-M3+CLIP fusion retrieval using NDCG@K."""

    # ── Load ESCI examples ─────────────────────────────────────────────────
    console.rule("[bold]Loading ESCI examples[/bold]")
    ex_path = ESCI_DIR / "shopping_queries_dataset_examples.parquet"
    examples = pd.read_parquet(ex_path)
    test_us = examples[(examples["split"] == split) & (examples["product_locale"] == "us")]
    console.print(f"  {split.capitalize()} US: {len(test_us):,} examples, "
                  f"{test_us['query_id'].nunique():,} queries")

    # Only keep queries that have at least one E or S label (otherwise NDCG is trivially 0)
    has_signal = (
        test_us.groupby("query_id")["esci_label"]
        .apply(lambda s: (s.isin(["E", "S"])).any())
    )
    valid_qids = has_signal[has_signal].index.tolist()
    test_us = test_us[test_us["query_id"].isin(valid_qids)]
    console.print(f"  Queries with E/S labels: {len(valid_qids):,}")

    # Sample n_queries
    random.seed(seed)
    sampled_qids = random.sample(valid_qids, min(n_queries, len(valid_qids)))
    test_us = test_us[test_us["query_id"].isin(sampled_qids)]
    console.print(f"  Sampled: {len(sampled_qids):,} queries for evaluation")

    # Build {query_id: query_text} and relevance index
    query_texts = (
        test_us.drop_duplicates("query_id")
        .set_index("query_id")["query"]
        .to_dict()
    )
    relevance_index = _build_relevance_index(test_us)

    # ── Initialise retrievers ──────────────────────────────────────────────
    console.rule("[bold]Initialising retrievers[/bold]")
    console.print("  Loading BGE-M3 retriever…")
    bge = ProductRetriever()

    clip = CLIPRetriever()
    clip_available = clip.available()
    if clip_available:
        console.print("  [green]CLIP collection available — will evaluate fusion[/green]")
        _ = clip.encode_text("warm up")
    else:
        console.print("  [yellow]CLIP collection not found — evaluating BGE-M3 only[/yellow]")

    console.print("  Loading cross-encoder reranker…")
    reranker = ProductReranker()
    _ = reranker.model   # warm up

    # ── Run evaluation ─────────────────────────────────────────────────────
    console.rule(f"[bold]Evaluating NDCG@{k} over {len(sampled_qids)} queries[/bold]")

    bge_result    = EvalResult(system="BGE-M3", k=k)
    fusion_result = EvalResult(system="BGE-M3 + CLIP (RRF)", k=k) if clip_available else None
    rerank_result = EvalResult(system="BGE-M3 + CLIP + Reranker", k=k) if clip_available else \
                    EvalResult(system="BGE-M3 + Reranker", k=k)

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(), MofNCompleteColumn(), TaskProgressColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(sampled_qids))

        for qid in sampled_qids:
            query = query_texts[str(qid)]
            relevance = relevance_index[str(qid)]

            # BGE-M3 only
            bge_hits = bge.retrieve(query, top_k=k)
            bge_ids  = [p.product_id for p in bge_hits]
            bge_result.scores.append(ndcg_at_k(bge_ids, relevance, k=k))

            # Fusion (BGE-M3 + CLIP)
            bge_wide = bge.retrieve(query, top_k=top_k_retrieve)
            if clip_available and fusion_result is not None:
                clip_wide  = clip.retrieve(query, top_k=top_k_retrieve)
                fused      = rrf_fuse([bge_wide, clip_wide], top_k=top_k_retrieve)
                fusion_ids = [p.product_id for p in fused[:k]]
                fusion_result.scores.append(ndcg_at_k(fusion_ids, relevance, k=k))
            else:
                fused = bge_wide

            # Reranker
            reranked     = reranker.rerank(query, fused, top_k=k)
            reranked_ids = [p.product_id for p in reranked]
            rerank_result.scores.append(ndcg_at_k(reranked_ids, relevance, k=k))

            progress.advance(task)

    # ── Results table ──────────────────────────────────────────────────────
    console.print()
    console.rule("[bold]Results[/bold]")

    table = Table(show_header=True, header_style="bold cyan", expand=False)
    table.add_column("System", min_width=28)
    table.add_column(f"NDCG@{k}", justify="right", min_width=10)
    table.add_column("Std Dev", justify="right", min_width=10)
    table.add_column("Queries", justify="right", min_width=8)

    def _fmt(result: EvalResult, highlight: bool = False) -> None:
        style = "bold green" if highlight else ""
        table.add_row(
            result.system,
            f"{result.mean:.4f}",
            f"± {result.std:.4f}",
            str(result.n),
            style=style,
        )

    _fmt(bge_result)
    if fusion_result:
        delta = fusion_result.mean - bge_result.mean
        _fmt(fusion_result, highlight=delta > 0)
        table.add_row("Delta (fusion − BGE-M3)", f"{delta:+.4f}", "—", "—",
                      style="bold green" if delta > 0 else "bold red")

    rerank_delta = rerank_result.mean - bge_result.mean
    _fmt(rerank_result, highlight=rerank_delta > 0)
    table.add_row("Delta (reranker − BGE-M3)", f"{rerank_delta:+.4f}", "—", "—",
                  style="bold green" if rerank_delta > 0 else "bold red")

    console.print(table)

    # ── Per-bucket breakdown ───────────────────────────────────────────────
    console.print()
    console.rule("[bold]BGE-M3 NDCG@10 distribution[/bold]")
    buckets = {"0.0–0.2": 0, "0.2–0.4": 0, "0.4–0.6": 0, "0.6–0.8": 0, "0.8–1.0": 0}
    for s in bge_result.scores:
        if s < 0.2:   buckets["0.0–0.2"] += 1
        elif s < 0.4: buckets["0.2–0.4"] += 1
        elif s < 0.6: buckets["0.4–0.6"] += 1
        elif s < 0.8: buckets["0.6–0.8"] += 1
        else:          buckets["0.8–1.0"] += 1

    dist_table = Table(show_header=True, header_style="bold", expand=False)
    dist_table.add_column("Score range")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("% of queries", justify="right")
    n = bge_result.n
    for bucket, count in buckets.items():
        dist_table.add_row(bucket, str(count), f"{count/n*100:.1f}%")
    console.print(dist_table)


if __name__ == "__main__":
    app()
