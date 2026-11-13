"""
Command Line Interface for CogentSearch Phase 3: LangGraph Agentic Loop.
Demonstrates the full self-correcting RAG loop:
Query Rewriter -> Retriever -> Document Grader -> (Conditional Loop / Web Fallback) -> Generator (Answer & Cite).
"""

import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cogent.agent.agentic_loop import AgenticRAG

console = Console()


def main():
    parser = argparse.ArgumentParser(description="CogentSearch Phase 3: LangGraph Agentic RAG CLI")
    parser.add_argument(
        "--query",
        type=str,
        default="What are the specific parameters governing financial protocol #1 in system specification FIN-10?",
        help="User question to submit to the Agentic RAG loop",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum rewrite retries before falling back to external web search",
    )
    args = parser.parse_args()

    console.print(Panel(f"[bold cyan]CogentSearch Phase 3: LangGraph Agentic Loop[/bold cyan]\nQuestion: [yellow]{args.query}[/yellow]"))

    agent = AgenticRAG(max_retries=args.max_retries)
    response = agent.run(question=args.query)

    # Display Execution Trace
    trace_table = Table(title="Execution Node Trace", show_header=True, header_style="bold magenta")
    trace_table.add_column("Step #", style="dim", width=8)
    trace_table.add_column("Node Executed", style="bold green")

    for idx, node_name in enumerate(response.trace, start=1):
        trace_table.add_row(str(idx), node_name)
    console.print(trace_table)

    # Display Query Rewriting & Strategy
    console.print(f"\n[bold]Original Query:[/bold] {response.question}")
    console.print(f"[bold]Rewritten Query:[/bold] [cyan]{response.rewritten_query}[/cyan]")
    console.print(f"[bold]Retrieved Chunks:[/bold] {response.retrieved_chunk_count}")
    console.print(f"[bold]Web Search Used:[/bold] {response.web_search_used}")
    console.print(f"[bold]Rewrite Loops:[/bold] {response.retry_count}")

    # Display Generated Answer
    console.print(Panel(f"[bold green]Synthesized Answer:[/bold green]\n\n{response.generation}", title="Generator Output"))

    # Display Extracted Citations
    if response.citations:
        cit_table = Table(title="Extracted & Verified Citations", show_header=True, header_style="bold blue")
        cit_table.add_column("#", style="dim", width=4)
        cit_table.add_column("Source Document", style="cyan")
        cit_table.add_column("Page", style="magenta", width=6)
        cit_table.add_column("Modality Type", style="yellow", width=14)
        cit_table.add_column("Full Citation Tag", style="white")

        for idx, cit in enumerate(response.citations, start=1):
            cit_table.add_row(
                str(idx),
                cit.source_file,
                str(cit.page_number),
                cit.modality_type,
                cit.citation_tag,
            )
        console.print(cit_table)
    else:
        console.print("[bold red]Warning: No citations extracted from generated output.[/bold red]")


if __name__ == "__main__":
    main()
