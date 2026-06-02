#!/usr/bin/env python3
"""
Download all datasets required for the local Rufus system.

Datasets:
  1. Amazon ESCI  — product search relevance (search ranking backbone)
  2. SQID         — ESCI + product images + pre-extracted CLIP embeddings
  3. Amazon Reviews 2023 — customer reviews for RAG grounding (Electronics subset)
  4. MG-ShopDial  — multi-goal conversational shopping dialogues (SIGIR 2023)
"""

import subprocess
import sys
from pathlib import Path

import typer
from datasets import load_dataset  # used by multiwoz fallback
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Download Rufus datasets.")
console = Console()

DATA_DIR = Path("data")

# Amazon Reviews subsets to download (keep manageable for dev)
REVIEW_CATEGORIES = [
    "raw_meta_Electronics",    # product metadata — needed for catalog RAG
    "raw_review_Electronics",  # customer reviews — needed for QA grounding
]


# ---------------------------------------------------------------------------
# Individual downloaders
# ---------------------------------------------------------------------------

def _already_exists(dest: Path) -> bool:
    return dest.exists() and any(dest.iterdir())


def download_esci(force: bool = False) -> bool:
    dest = DATA_DIR / "esci"
    if not force and _already_exists(dest):
        console.print("[yellow]  ESCI already downloaded — skipping (use --force to re-download)[/yellow]")
        return True

    console.print("  Cloning amazon-science/esci-data from GitHub …")
    try:
        subprocess.run(
            [
                "git", "clone", "--depth=1",
                "https://github.com/amazon-science/esci-data.git",
                str(dest),
            ],
            check=True,
            capture_output=False,
        )
        console.print("  [green]ESCI done[/green]")
        return True
    except subprocess.CalledProcessError as exc:
        console.print(f"  [red]ESCI failed: {exc}[/red]")
        return False


def download_sqid(force: bool = False) -> bool:
    dest = DATA_DIR / "sqid"
    if not force and _already_exists(dest):
        console.print("[yellow]  SQID already downloaded — skipping[/yellow]")
        return True

    console.print("  Downloading crossingminds/shopping-queries-image-dataset from HF Hub …")
    console.print("  [dim](includes pre-extracted CLIP embeddings — ~several GB)[/dim]")
    try:
        snapshot_download(
            repo_id="crossingminds/shopping-queries-image-dataset",
            repo_type="dataset",
            local_dir=str(dest),
        )
        console.print("  [green]SQID done[/green]")
        return True
    except Exception as exc:
        console.print(f"  [red]SQID failed: {exc}[/red]")
        return False


def download_amazon_reviews(force: bool = False) -> bool:
    # Download metadata only for Week 1 (product catalog RAG).
    # raw_review_Electronics is ~22 GB — run `amazon-reviews --reviews` when
    # you have enough free disk space.
    dest = DATA_DIR / "amazon_reviews"
    if not force and _already_exists(dest):
        console.print("[yellow]  Amazon Reviews already downloaded — skipping[/yellow]")
        return True

    console.print("  Downloading McAuley-Lab/Amazon-Reviews-2023 — metadata only …")
    console.print("  [dim](raw_meta_Electronics — product catalog for RAG, ~1–2 GB)[/dim]")
    try:
        snapshot_download(
            repo_id="McAuley-Lab/Amazon-Reviews-2023",
            repo_type="dataset",
            allow_patterns=["*meta_Electronics*"],
            local_dir=str(dest),
        )
        console.print("  [green]Amazon Reviews (metadata) done[/green]")
        return True
    except Exception as exc:
        console.print(f"  [red]Amazon Reviews failed: {exc}[/red]")
        return False


def download_amazon_reviews_full(force: bool = False) -> bool:
    """Download full Electronics reviews (~22 GB). Requires ~25 GB free space."""
    dest = DATA_DIR / "amazon_reviews"
    console.print("  Downloading McAuley-Lab/Amazon-Reviews-2023 — full Electronics (reviews + meta) …")
    console.print("  [dim](~22 GB — make sure you have ~25 GB free)[/dim]")
    try:
        snapshot_download(
            repo_id="McAuley-Lab/Amazon-Reviews-2023",
            repo_type="dataset",
            allow_patterns=["*Electronics*"],
            local_dir=str(dest),
        )
        console.print("  [green]Amazon Reviews (full) done[/green]")
        return True
    except Exception as exc:
        console.print(f"  [red]Amazon Reviews (full) failed: {exc}[/red]")
        return False


def download_mgshop_dial(force: bool = False) -> bool:
    dest = DATA_DIR / "mgshop_dial"
    if not force and _already_exists(dest):
        console.print("[yellow]  MG-ShopDial already downloaded — skipping[/yellow]")
        return True

    console.print("  Cloning iai-group/MG-ShopDial from GitHub …")
    try:
        subprocess.run(
            [
                "git", "clone", "--depth=1",
                "https://github.com/iai-group/MG-ShopDial.git",
                str(dest),
            ],
            check=True,
            capture_output=False,
        )
        console.print("  [green]MG-ShopDial done[/green]")
        return True
    except subprocess.CalledProcessError as exc:
        console.print(f"  [red]MG-ShopDial failed: {exc}[/red]")
        return False


def download_multiwoz(force: bool = False) -> bool:
    """Fallback conversational dataset if MG-ShopDial is unavailable."""
    dest = DATA_DIR / "multiwoz"
    if not force and _already_exists(dest):
        console.print("[yellow]  MultiWOZ already downloaded — skipping[/yellow]")
        return True

    console.print("  Downloading MultiWOZ 2.2 (fallback for dialogue fine-tuning) …")
    try:
        ds = load_dataset("multi_woz_v22", trust_remote_code=True)
        ds.save_to_disk(str(dest))
        console.print("  [green]MultiWOZ done[/green]")
        return True
    except Exception as exc:
        console.print(f"  [red]MultiWOZ failed: {exc}[/red]")
        return False


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "esci":           ("Amazon ESCI Shopping Queries",          download_esci),
    "sqid":           ("SQID (Shopping Queries + Images)",       download_sqid),
    "amazon_reviews": ("Amazon Reviews 2023 — Electronics",      download_amazon_reviews),
    "mgshop_dial":    ("MG-ShopDial (Conversational Shopping)",  download_mgshop_dial),
    "multiwoz":       ("MultiWOZ 2.2 (dialogue fallback)",       download_multiwoz),
}


@app.command("all")
def cmd_all(force: bool = typer.Option(False, "--force", help="Re-download even if present")):
    """Download all Rufus datasets (except MultiWOZ fallback)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    for key in ["esci", "sqid", "amazon_reviews", "mgshop_dial"]:
        name, fn = DOWNLOADERS[key]
        console.rule(f"[bold]{name}[/bold]")
        results[name] = fn(force=force)
        console.print()

    _print_summary(results)


@app.command("esci")
def cmd_esci(force: bool = typer.Option(False, "--force")):
    """Download Amazon ESCI dataset only."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_esci(force=force)


@app.command("sqid")
def cmd_sqid(force: bool = typer.Option(False, "--force")):
    """Download SQID dataset only."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_sqid(force=force)


@app.command("amazon-reviews")
def cmd_reviews(
    force: bool = typer.Option(False, "--force"),
    reviews: bool = typer.Option(False, "--reviews", help="Also download full reviews (~22 GB, needs ~25 GB free)"),
):
    """Download Amazon Reviews 2023 — metadata by default, add --reviews for full dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if reviews:
        download_amazon_reviews_full(force=force)
    else:
        download_amazon_reviews(force=force)


@app.command("mgshop-dial")
def cmd_mgshop(force: bool = typer.Option(False, "--force")):
    """Write MG-ShopDial download instructions."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_mgshop_dial(force=force)


@app.command("multiwoz")
def cmd_multiwoz(force: bool = typer.Option(False, "--force")):
    """Download MultiWOZ 2.2 (fallback for dialogue fine-tuning)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_multiwoz(force=force)


def _print_summary(results: dict[str, bool]) -> None:
    table = Table(title="Download Summary", show_header=True, header_style="bold")
    table.add_column("Dataset", min_width=40)
    table.add_column("Status", min_width=10)

    for name, ok in results.items():
        status = "[green]✓  Done[/green]" if ok else "[red]✗  Failed[/red]"
        table.add_row(name, status)

    console.print(table)


if __name__ == "__main__":
    app()
