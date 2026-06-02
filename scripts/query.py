#!/usr/bin/env python3
"""
Interactive Rufus query CLI.

Usage:
  uv run python scripts/query.py                         # REPL
  uv run python scripts/query.py --q "best wifi router"  # single query
  uv run python scripts/query.py --top-k 8               # more products
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from rufus.rag import RufusRAG

app = typer.Typer(help="Query Rufus.")
console = Console()


def _print_products(products) -> None:
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("#", width=3)
    table.add_column("Score", width=6)
    table.add_column("Title")
    table.add_column("Brand", width=20)

    for i, p in enumerate(products, 1):
        table.add_row(
            str(i),
            f"{p.score:.3f}",
            p.title[:70] + ("…" if len(p.title) > 70 else ""),
            p.brand or "—",
        )
    console.print(table)


def _run_query(rufus: RufusRAG, question: str) -> None:
    console.print(Rule(f"[bold]Query:[/bold] {question}"))

    with console.status("Retrieving products…", spinner="dots"):
        products, answer_iter = rufus.query(question, stream=True)

    console.print("\n[bold cyan]Retrieved Products[/bold cyan]")
    _print_products(products)

    console.print("\n[bold cyan]Rufus Answer[/bold cyan]")
    full_answer = ""
    for chunk in answer_iter:
        console.print(chunk, end="")
        full_answer += chunk
    console.print("\n")


@app.command()
def main(
    q: str = typer.Option("", "--q", "-q", help="Single question (omit for REPL)"),
    top_k: int = typer.Option(5, "--top-k", help="Number of products to retrieve"),
    model: str = typer.Option("qwen3.5:latest", "--model", help="Ollama model"),
):
    rufus = RufusRAG(ollama_model=model, top_k=top_k)

    if q:
        _run_query(rufus, q)
        return

    # REPL mode
    console.print(Panel(
        "[bold green]Rufus[/bold green] — local shopping assistant\n"
        "Type your question and press Enter. Type [bold]exit[/bold] to quit.",
        expand=False,
    ))

    while True:
        try:
            question = console.input("\n[bold yellow]You:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        _run_query(rufus, question)


if __name__ == "__main__":
    app()
