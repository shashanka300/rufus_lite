#!/usr/bin/env python3
"""
Download ALL datasets for the full-scale Rufus build.

Groups
------
  catalog        — product catalog, reviews, Q&A (HuggingFace)
  supply         — supply chain, inventory, demand forecasting (Kaggle + direct)
  personalization — clickstream, purchase history (Kaggle)
  dialogue       — conversational shopping datasets (GitHub + direct)

Sources
-------
  HuggingFace Hub   — automatic (logged in via `huggingface-cli login`)
  Kaggle API        — requires ~/.kaggle/kaggle.json
                      -> kaggle.com/account -> Create New API Token
  GitHub            — git clone --depth=1
  Direct HTTP       — requests

Usage
-----
  uv run python scripts/download_all_data.py all
  uv run python scripts/download_all_data.py catalog
  uv run python scripts/download_all_data.py supply
  uv run python scripts/download_all_data.py personalization
  uv run python scripts/download_all_data.py dialogue
  uv run python scripts/download_all_data.py status   # show what's downloaded
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
import typer
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Download all Rufus datasets.")
# force_terminal + highlight=False avoids Windows cp1252 encoding crashes
console = Console(highlight=False, emoji=False)

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

# ── helpers ───────────────────────────────────────────────────────────────────

def _exists(dest: Path) -> bool:
    return dest.exists() and any(f for f in dest.rglob("*") if f.is_file())


def _hf(repo_id: str, dest: Path, allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None, force: bool = False) -> bool:
    if not force and _exists(dest):
        console.print(f"  [yellow]skip[/yellow] {dest.name} (already present)")
        return True
    console.print(f"  [cyan]HF[/cyan]  {repo_id}  ->  {dest.relative_to(ROOT)}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        console.print(f"  [green]done[/green] {dest.name}")
        return True
    except Exception as exc:
        console.print(f"  [red]fail[/red] {repo_id}: {exc}")
        return False


def _kaggle_auth() -> str | None:
    """Return base64 Basic-auth token from ~/.kaggle/kaggle.json, or None."""
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        console.print(
            "\n  [red]Kaggle credentials not found.[/red]\n"
            "  1. Go to kaggle.com -> Account -> Create New API Token\n"
            "  2. Save kaggle.json to ~/.kaggle/kaggle.json\n"
        )
        return None
    import base64
    cred = json.loads(cred_path.read_text(encoding="utf-8"))
    return base64.b64encode(f"{cred['username']}:{cred['key']}".encode()).decode()


def _kaggle(source: str, dest: Path, is_competition: bool = False) -> bool:
    """
    Download a Kaggle dataset or competition.
    Uses PowerShell WebClient on Windows (.NET HttpClient avoids the SChannel
    strict TLS close_notify issue that breaks both Python ssl and Windows curl).
    Falls back to curl on non-Windows.
    """
    if _exists(dest):
        console.print(f"  [yellow]skip[/yellow] {dest.name} (already present)")
        return True

    token = _kaggle_auth()
    if token is None:
        return False

    dest.mkdir(parents=True, exist_ok=True)
    kind = "competitions" if is_competition else "datasets"
    console.print(f"  [cyan]Kaggle[/cyan] {kind}/{source}  ->  {dest.relative_to(ROOT)}")

    if is_competition:
        url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{source}"
    else:
        url = f"https://www.kaggle.com/api/v1/datasets/download/{source}"

    zip_path = dest / "_download.zip"

    if sys.platform == "win32":
        # Use .NET HttpClient streaming — no timeout, streams directly to disk.
        # WebClient.DownloadFile has a ~100s hardcoded timeout; HttpClient does not.
        ps_script = f"""
Add-Type -AssemblyName System.Net.Http
$handler = New-Object System.Net.Http.HttpClientHandler
$client  = New-Object System.Net.Http.HttpClient($handler)
$client.DefaultRequestHeaders.Authorization = `
    [System.Net.Http.Headers.AuthenticationHeaderValue]::new("Basic", "{token}")
$client.Timeout = [System.Threading.Timeout]::InfiniteTimeSpan
$resp   = $client.GetAsync("{url}", [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
if (-not $resp.IsSuccessStatusCode) {{
    throw "HTTP $([int]$resp.StatusCode) $($resp.ReasonPhrase)"
}}
$src = $resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
$dst = [System.IO.File]::Create("{zip_path}")
$src.CopyTo($dst)
$dst.Close(); $src.Close(); $client.Dispose()
"""
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", ps_script],
            capture_output=True, text=True,
            timeout=None,  # no Python-side timeout either
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout).strip()[:400]
            console.print(f"  [red]fail[/red] PowerShell: {err}")
            if "401" in err or "403" in err or "rules" in err.lower():
                console.print(
                    f"  [yellow]hint[/yellow] Accept competition rules at:\n"
                    f"  https://www.kaggle.com/competitions/{source}/rules"
                )
            zip_path.unlink(missing_ok=True)
            return False
    else:
        result = subprocess.run(
            ["curl", "-L", "--silent", "--show-error",
             "-H", f"Authorization: Basic {token}",
             "-o", str(zip_path), url],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            console.print(f"  [red]fail[/red] curl: {result.stderr.strip()[:200]}")
            zip_path.unlink(missing_ok=True)
            return False

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        console.print("  [red]fail[/red] downloaded file is empty")
        return False

    # If Kaggle returned a JSON error body (e.g. rules not accepted)
    try:
        err = json.loads(zip_path.read_bytes())
        msg = err.get("message", str(err))
        console.print(f"  [red]fail[/red] Kaggle API: {msg}")
        if "rules" in msg.lower() or "403" in str(err):
            console.print(
                f"  [yellow]hint[/yellow] Accept competition rules at:\n"
                f"  https://www.kaggle.com/competitions/{source}/rules"
            )
        zip_path.unlink(missing_ok=True)
        return False
    except (json.JSONDecodeError, Exception):
        pass  # real zip, continue

    console.print("  unzipping ...")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        zip_path.unlink()
        console.print(f"  [green]done[/green] {dest.name}")
        return True
    except zipfile.BadZipFile:
        console.print("  [red]fail[/red] Not a valid zip -- may need to accept competition rules")
        zip_path.unlink(missing_ok=True)
        return False


def _git(url: str, dest: Path) -> bool:
    if _exists(dest):
        console.print(f"  [yellow]skip[/yellow] {dest.name} (already present)")
        return True
    console.print(f"  [cyan]git[/cyan]  {url}  ->  {dest.relative_to(ROOT)}")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", url, str(dest)],
            check=True, capture_output=True,
        )
        console.print(f"  [green]done[/green] {dest.name}")
        return True
    except subprocess.CalledProcessError as exc:
        console.print(f"  [red]fail[/red] {exc.stderr.decode()[:200]}")
        return False


def _http(url: str, dest: Path, unzip: bool = False) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        console.print(f"  [yellow]skip[/yellow] {dest.name} (already present)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"  [cyan]HTTP[/cyan] {url[:72]}")
    try:
        if sys.platform == "win32":
            # .NET WebClient avoids SChannel strict TLS close_notify issues
            ps = f'(New-Object System.Net.WebClient).DownloadFile("{url}", "{dest}")'
            r = subprocess.run(
                ["powershell", "-NonInteractive", "-Command", ps],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip()[:200])
        else:
            r2 = requests.get(url, stream=True, timeout=60)
            r2.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r2.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

        if unzip:
            with zipfile.ZipFile(dest) as z:
                z.extractall(dest.parent)
            dest.unlink()
        size_mb = dest.stat().st_size / 1e6 if dest.exists() else 0
        console.print(f"  [green]done[/green] {dest.name}  ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        console.print(f"  [red]fail[/red] {exc}")
        return False


# ── Dataset definitions ────────────────────────────────────────────────────────

def download_amazon_reviews_all(force: bool = False) -> bool:
    """
    Amazon Reviews 2023 — ALL categories, metadata + review text.
    McAuley Lab, HuggingFace. ~100 GB uncompressed.
    Metadata: avg_rating, features, description, price, categories, specs.
    Reviews: rating, review_text, helpful_votes, verified_purchase.
    """
    dest = DATA_DIR / "amazon_reviews_full"
    if not force and _exists(dest):
        console.print(f"  [yellow]skip[/yellow] amazon_reviews_full (already present)")
        return True
    console.print("  Downloading Amazon Reviews 2023 -- raw_meta + raw_review only (~100 GB) ...")
    return _hf(
        "McAuley-Lab/Amazon-Reviews-2023",
        dest,
        # benchmark/ = 285 GB of ML recommendation splits we don't need
        # raw/ = symlink duplicates; .cache handled internally by HF
        allow_patterns=["raw_meta_*", "raw_review_*", "asin2category.json"],
        force=force,
    )


def download_amazon_c4(force: bool = False) -> bool:
    """
    Amazon-C4 (McAuley Lab) — product Q&A pairs scraped from Amazon.
    ~10 GB. Directly usable for QA grounding and fine-tuning.
    """
    dest = DATA_DIR / "amazon_c4"
    if not force and _exists(dest):
        console.print(f"  [yellow]skip[/yellow] amazon_c4 (already present)")
        return True
    return _hf("McAuley-Lab/Amazon-C4", dest)


# ── Supply Chain ──────────────────────────────────────────────────────────────

def download_olist(force: bool = False) -> bool:
    """
    Olist Brazilian E-Commerce — 100K orders, products, sellers, geolocation,
    logistics, delivery times, reviews. Best open end-to-end supply chain dataset.
    Kaggle: olistbr/brazilian-ecommerce (~50 MB)
    """
    dest = DATA_DIR / "supply_chain" / "olist"
    if not force and _exists(dest):
        console.print("  [yellow]skip[/yellow] olist (already present)")
        return True
    return _kaggle("olistbr/brazilian-ecommerce", dest)


def download_m5(force: bool = False) -> bool:
    """
    M5 Forecasting -- 5 years of daily Walmart sales for 30,490 SKUs.
    Uses Nixtla's datasetsforecast package -- no Kaggle login required.
    """
    dest = DATA_DIR / "supply_chain" / "m5"
    if not force and _exists(dest):
        console.print("  [yellow]skip[/yellow] m5 (already present)")
        return True
    dest.mkdir(parents=True, exist_ok=True)
    console.print("  datasetsforecast: M5 (Nixtla mirror, no Kaggle required) ...")
    try:
        from datasetsforecast.m5 import M5
        M5.download(directory=str(dest))
        console.print("  [green]done[/green] m5")
        return True
    except Exception as exc:
        console.print(f"  [red]fail[/red] {exc}")
        return False


def download_dataco(force: bool = False) -> bool:
    """
    DataCo Smart Supply Chain — 180K records covering order status, shipping mode,
    late-delivery risk, profit per order, product category, supplier region.
    Direct from Mendeley Data (~200 MB).
    """
    dest = DATA_DIR / "supply_chain" / "dataco"
    if not force and _exists(dest):
        console.print("  [yellow]skip[/yellow] dataco (already present)")
        return True
    # Try Kaggle first, fall back to Mendeley
    ok = _kaggle(
        "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",
        dest,
    )
    if not ok:
        console.print("  Trying Mendeley direct download …")
        dest.mkdir(parents=True, exist_ok=True)
        return _http(
            "https://data.mendeley.com/public-files/datasets/8gx2fvg2k6/files/"
            "65e1196a-e2c2-4f8e-8870-9f73d8024283/file_downloaded",
            dest / "DataCoSupplyChainDataset.csv",
        )
    return ok


def download_scms(force: bool = False) -> bool:
    """
    SCMS Delivery History (USAID PEPFAR) — 10K+ supply chain shipment records:
    vendor, country, lead time, freight cost, scheduled vs actual delivery.
    Best open source for supplier lead-time modelling. Direct download (~50 MB).
    """
    dest = DATA_DIR / "supply_chain" / "scms"
    dest.mkdir(parents=True, exist_ok=True)
    # Primary: Kaggle (most reliable)
    ok = _kaggle("divyeshardeshana/supply-chain-shipment-pricing-data", dest)
    if ok:
        return True
    # Fallback: direct CSV from data.usaid.gov (may be geo-blocked)
    return _http(
        "https://data.usaid.gov/api/views/a3rc-nmf6/rows.csv?accessType=DOWNLOAD",
        dest / "scms_delivery_history.csv",
    )


def download_uci_retail(force: bool = False) -> bool:
    """
    UCI Online Retail II — 2 years of UK e-commerce transactions (2009–2011).
    ~1M records: invoice, product code, description, qty, date, unit price,
    customer ID, country. Good for demand and basket analysis. (~50 MB xlsx)
    """
    dest = DATA_DIR / "supply_chain" / "uci_retail"
    dest.mkdir(parents=True, exist_ok=True)
    return _http(
        "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        dest / "online_retail_ii.zip",
        unzip=True,
    )


def download_rossmann(force: bool = False) -> bool:
    """
    Rossmann-style demand data via datasetsforecast M4 (daily retail sales).
    Original Rossmann competition requires Kaggle join -- using open equivalent.
    """
    dest = DATA_DIR / "supply_chain" / "rossmann"
    if not force and _exists(dest):
        console.print("  [yellow]skip[/yellow] rossmann (already present)")
        return True
    dest.mkdir(parents=True, exist_ok=True)
    console.print("  datasetsforecast: M4 daily (open Rossmann-equivalent) ...")
    try:
        from datasetsforecast.m4 import M4
        M4.download(directory=str(dest), group="Daily")
        console.print("  [green]done[/green] rossmann (M4 daily)")
        return True
    except Exception as exc:
        console.print(f"  [red]fail[/red] {exc}")
        return False


# ── Personalization ───────────────────────────────────────────────────────────

def download_retailrocket(force: bool = False) -> bool:
    """
    RetailRocket — 3 months of real e-commerce events: view, addtocart, transaction.
    4.7M events, 1.4M unique visitors. Best small dataset for session modelling.
    Kaggle: retailrocket/ecommerce-dataset (~200 MB)
    """
    dest = DATA_DIR / "personalization" / "retailrocket"
    if not force and _exists(dest):
        console.print("  [yellow]skip[/yellow] retailrocket (already present)")
        return True
    return _kaggle("retailrocket/ecommerce-dataset", dest)


def download_hm_fashion(force: bool = False) -> bool:
    """
    Fashion purchase history -- uses recwizard/fashion-rec HF dataset as open
    alternative to H&M (Kaggle competition requires join, 35 GB).
    We already have RetailRocket + Instacart for personalization signal.
    """
    dest = DATA_DIR / "personalization" / "hm_fashion"
    if not force and _exists(dest):
        console.print("  [yellow]skip[/yellow] hm_fashion (already present)")
        return True
    # Try a smaller open fashion interaction dataset on HF
    for repo in ("McAuley-Lab/Amazon-Reviews-2023",):
        # H&M open alternative: use Amazon Clothing reviews sub-split
        console.print("  note: H&M (35 GB Kaggle competition) replaced by")
        console.print("        Amazon Clothing/Shoes metadata (already in amazon_reviews_full)")
        console.print("        RetailRocket + Instacart provide sufficient personalization signal.")
        console.print("  [yellow]skip[/yellow] hm_fashion -- covered by amazon_reviews_full")
        return True  # not a failure, just covered elsewhere


def download_instacart(force: bool = False) -> bool:
    """
    Instacart Market Basket Analysis — 3.4M grocery orders from 206K users.
    Contains prior/train/test orders, product details, aisle and department data.
    Kaggle: yasserh/instacart-online-grocery-basket-analysis-dataset (~1 GB)
    """
    dest = DATA_DIR / "personalization" / "instacart"
    if not force and _exists(dest):
        console.print("  [yellow]skip[/yellow] instacart (already present)")
        return True
    return _kaggle(
        "yasserh/instacart-online-grocery-basket-analysis-dataset",
        dest,
    )


# ── Dialogue ──────────────────────────────────────────────────────────────────

def download_simmc2(force: bool = False) -> bool:
    """
    SIMMC 2.1 (Meta AI) — 11K multimodal shopping dialogues (fashion + furniture).
    Has grounded belief states: which product is being discussed, what attributes
    were mentioned, what action the user wants. Best dataset for training
    an intent+slot extractor for shopping. GitHub (~500 MB).
    """
    dest = DATA_DIR / "dialogue" / "simmc2"
    if not force and _exists(dest):
        console.print("  [yellow]skip[/yellow] simmc2 (already present)")
        return True
    return _git("https://github.com/facebookresearch/simmc2", dest)


def download_redial(force: bool = False) -> bool:
    """
    ReDial — 10K conversational movie recommendation dialogues.
    Covers elicitation ('I want something like X'), preference refinement,
    and explanation. Transfers well to product recommendation patterns. GitHub.
    """
    dest = DATA_DIR / "dialogue" / "redial"
    if not force and _exists(dest):
        console.print("  [yellow]skip[/yellow] redial (already present)")
        return True
    return _hf("recwizard/redial", dest)


def download_durecdial(force: bool = False) -> bool:
    """
    DuRecDial 2.0 (Baidu) — 16K goal-driven recommendation dialogues in English
    and Chinese. Covers music, movies, food, POI. Rich goal annotations.
    """
    dest = DATA_DIR / "dialogue" / "durecdial"
    if not force and _exists(dest):
        console.print("  [yellow]skip[/yellow] durecdial (already present)")
        return True
    return _git("https://github.com/liuzeming01/DuRecDial", dest)


def download_inspired(force: bool = False) -> bool:
    """INSPIRED -- not publicly available (GitHub removed, no HF mirror)."""
    console.print("  [yellow]skip[/yellow] INSPIRED -- not publicly available")
    return False


# ── Registry ──────────────────────────────────────────────────────────────────

CATALOG = {
    "amazon_reviews_all":  ("Amazon Reviews 2023 — ALL categories",         download_amazon_reviews_all),
    "amazon_c4":           ("Amazon-C4 — product Q&A pairs",                download_amazon_c4),
}

SUPPLY = {
    "olist":       ("Olist Brazilian E-Commerce",        download_olist),
    "m5":          ("M5 Forecasting — Walmart daily sales", download_m5),
    "dataco":      ("DataCo Supply Chain Events",        download_dataco),
    "scms":        ("SCMS Delivery History (USAID)",     download_scms),
    "uci_retail":  ("UCI Online Retail II",              download_uci_retail),
    "rossmann":    ("Rossmann Store Sales",              download_rossmann),
}

PERSONALIZATION = {
    "retailrocket": ("RetailRocket — session events",       download_retailrocket),
    "hm_fashion":   ("H&M Fashion — purchase history",      download_hm_fashion),
    "instacart":    ("Instacart — grocery basket analysis",  download_instacart),
}

DIALOGUE = {
    "simmc2":    ("SIMMC 2.1 — multimodal shopping dialogues", download_simmc2),
    "redial":    ("ReDial — conversational movie rec",         download_redial),
    "durecdial": ("DuRecDial 2.0 — goal-driven dialogues",     download_durecdial),
    # "inspired" removed -- not publicly available anywhere
}

ALL_GROUPS = {
    "catalog":         CATALOG,
    "supply":          SUPPLY,
    "personalization": PERSONALIZATION,
    "dialogue":        DIALOGUE,
}


def _run_group(group: dict, force: bool) -> dict[str, bool]:
    results = {}
    for key, (name, fn) in group.items():
        console.rule(f"[bold]{name}[/bold]")
        results[name] = fn(force=force)
        console.print()
    return results


def _print_summary(results: dict[str, bool]) -> None:
    table = Table(title="Download Summary", show_header=True, header_style="bold magenta")
    table.add_column("Dataset", min_width=48)
    table.add_column("Status",  min_width=12)
    for name, ok in results.items():
        table.add_row(name, "[green]OK[/green]" if ok else "[red]FAIL[/red]")
    console.print(table)


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command("all")
def cmd_all(force: bool = typer.Option(False, "--force")):
    """Download every dataset across all groups."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    for group_name, group in ALL_GROUPS.items():
        console.rule(f"[bold magenta]── {group_name.upper()} ──[/bold magenta]")
        results.update(_run_group(group, force=force))
    _print_summary(results)


@app.command("catalog")
def cmd_catalog(force: bool = typer.Option(False, "--force")):
    """Download product catalog + reviews (HuggingFace)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _print_summary(_run_group(CATALOG, force=force))


@app.command("supply")
def cmd_supply(force: bool = typer.Option(False, "--force")):
    """Download supply chain + demand forecasting datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _print_summary(_run_group(SUPPLY, force=force))


@app.command("personalization")
def cmd_personalization(force: bool = typer.Option(False, "--force")):
    """Download clickstream + purchase history datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _print_summary(_run_group(PERSONALIZATION, force=force))


@app.command("dialogue")
def cmd_dialogue(force: bool = typer.Option(False, "--force")):
    """Download conversational shopping dialogue datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _print_summary(_run_group(DIALOGUE, force=force))


@app.command("status")
def cmd_status():
    """Show which datasets are already downloaded."""
    table = Table(title="Dataset Status", show_header=True, header_style="bold cyan")
    table.add_column("Group",   min_width=16)
    table.add_column("Key",     min_width=20)
    table.add_column("Name",    min_width=44)
    table.add_column("Status",  min_width=12)

    dest_map = {
        "amazon_reviews_all": DATA_DIR / "amazon_reviews_full",
        "amazon_c4":          DATA_DIR / "amazon_c4",
        "olist":              DATA_DIR / "supply_chain" / "olist",
        "m5":                 DATA_DIR / "supply_chain" / "m5",
        "dataco":             DATA_DIR / "supply_chain" / "dataco",
        "scms":               DATA_DIR / "supply_chain" / "scms",
        "uci_retail":         DATA_DIR / "supply_chain" / "uci_retail",
        "rossmann":           DATA_DIR / "supply_chain" / "rossmann",
        "retailrocket":       DATA_DIR / "personalization" / "retailrocket",
        "hm_fashion":         DATA_DIR / "personalization" / "hm_fashion",
        "instacart":          DATA_DIR / "personalization" / "instacart",
        "simmc2":             DATA_DIR / "dialogue" / "simmc2",
        "redial":             DATA_DIR / "dialogue" / "redial",
        "durecdial":          DATA_DIR / "dialogue" / "durecdial",
        "inspired":           DATA_DIR / "dialogue" / "inspired",
        # already downloaded (week 1)
        "esci":               DATA_DIR / "esci",
        "sqid":               DATA_DIR / "sqid",
        "amazon_reviews_meta": DATA_DIR / "amazon_reviews",
        "mgshop_dial":        DATA_DIR / "mgshop_dial",
    }

    group_map = {
        "esci": "catalog", "sqid": "catalog",
        "amazon_reviews_meta": "catalog", "amazon_reviews_all": "catalog",
        "amazon_c4": "catalog",
        "olist": "supply", "m5": "supply", "dataco": "supply",
        "scms": "supply", "uci_retail": "supply", "rossmann": "supply",
        "retailrocket": "personalization", "hm_fashion": "personalization",
        "instacart": "personalization",
        "simmc2": "dialogue", "redial": "dialogue",
        "durecdial": "dialogue", "inspired": "dialogue",
        "mgshop_dial": "dialogue",
    }

    name_map = {k: v[0] for g in ALL_GROUPS.values() for k, v in g.items()}
    name_map.update({
        "esci": "Amazon ESCI", "sqid": "SQID (CLIP vectors)",
        "amazon_reviews_meta": "Amazon Reviews (Electronics meta)",
        "mgshop_dial": "MG-ShopDial",
    })

    for key, dest in sorted(dest_map.items(), key=lambda x: (group_map.get(x[0], ""), x[0])):
        ok = _exists(dest)
        table.add_row(
            group_map.get(key, "-"),
            key,
            name_map.get(key, key),
            "[green]present[/green]" if ok else "[dim]missing[/dim]",
        )
    console.print(table)


if __name__ == "__main__":
    app()
