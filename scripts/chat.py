#!/usr/bin/env python3
"""
Multi-turn conversational Rufus CLI.

Usage:
  uv run python scripts/chat.py                        # REPL (auto session)
  uv run python scripts/chat.py --model qwen3.5:27b    # different model
  uv run python scripts/chat.py --session abc123       # resume named session
"""

import uuid

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from rufus.graph import build_graph

app = typer.Typer(help="Multi-turn Rufus shopping assistant.")
console = Console()


def _print_products(products: list) -> None:
    if not products:
        return
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("#", width=3)
    table.add_column("Score", width=6)
    table.add_column("Title")
    table.add_column("Brand", width=18)
    for i, p in enumerate(products, 1):
        table.add_row(
            str(i),
            f"{p.score:.3f}",
            p.title[:72] + ("…" if len(p.title) > 72 else ""),
            p.brand or "—",
        )
    console.print(table)


@app.command()
def main(
    model: str = typer.Option("qwen3.5:latest", "--model", "-m", help="Ollama model for generation"),
    session: str = typer.Option("", "--session", "-s", help="Session ID (auto-generated if omitted)"),
):
    """Start a multi-turn shopping conversation with Rufus."""
    rufus = build_graph()
    thread_id = session or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id, "model": model}}

    console.print(Panel(
        f"[bold green]Rufus[/bold green] — multi-turn shopping assistant  "
        f"[dim](model: {model})[/dim]\n"
        f"Session: [dim]{thread_id[:16]}…[/dim]\n\n"
        "Ask anything — I remember the conversation. "
        "Type [bold]exit[/bold] to quit.",
        title="[bold]Welcome[/bold]",
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
        if question.lower() in {"exit", "quit", "q", "bye"}:
            console.print("[dim]Goodbye![/dim]")
            break

        with console.status("[dim]Thinking…[/dim]", spinner="dots"):
            result = rufus.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )

        intent = result.get("intent", "?")
        filters = result.get("filters") or {}
        active_filters = {k: v for k, v in filters.items() if v is not None}
        filter_str = f"  filters: {active_filters}" if active_filters else ""
        console.print(Rule(f"[dim]intent: {intent}{filter_str}[/dim]"))

        products = result.get("products") or []
        if products and intent in ("search", "qa", "compare"):
            console.print("\n[bold cyan]Retrieved Products[/bold cyan]")
            _print_products(products)

        answer = result["messages"][-1]["content"]
        console.print(f"\n[bold green]Rufus:[/bold green] {answer}")


if __name__ == "__main__":
    app()
