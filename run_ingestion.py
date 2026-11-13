"""
CLI Entry Point for CogentSearch Multimodal Ingestion Pipeline.
Usage:
    python run_ingestion.py --file path/to/document.pdf
    python run_ingestion.py --file path/to/presentation.pptx
    python run_ingestion.py --file path/to/report.docx
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from cogent.core.config import settings
from cogent.ingestion.pipeline import MultimodalIngestionPipeline


console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="CogentSearch: Multimodal Ingestion & Vision Enrichment Pipeline"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to PDF, DOCX, or PPTX document to ingest.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Target text chunk size in characters.",
    )
    parser.add_argument(
        "--no-vision-injection",
        action="store_true",
        help="Disable injecting Vision LLM summaries into retrieval chunks.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "mock"],
        default=settings.VISION_MODEL_PROVIDER,
        help="Vision LLM provider to use for visual reasoning.",
    )

    args = parser.parse_args()
    target_path = Path(args.file)

    if not target_path.exists():
        console.print(f"[bold red]Error:[/bold red] Target document '{target_path}' does not exist.")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold cyan]CogentSearch Ingestion Engine[/bold cyan]\n"
            f"File: [yellow]{target_path.name}[/yellow]\n"
            f"Type: [green]{target_path.suffix.upper()}[/green]\n"
            f"Vision Provider: [magenta]{args.provider}[/magenta]",
            title="[bold blue]Phase 1: Advanced Multimodal Ingestion[/bold blue]",
            expand=False,
        )
    )

    pipeline = MultimodalIngestionPipeline(
        chunk_size=args.chunk_size,
        inject_visual_summaries=not args.no_vision_injection,
    )

    doc = pipeline.process_file(target_path)

    # Display Metrics Table
    stats = doc.get_summary_stats()
    stats_table = Table(title="Document Ingestion Summary", show_header=True, header_style="bold green")
    stats_table.add_column("Metric", style="dim", width=25)
    stats_table.add_column("Value", style="bold")

    for k, v in stats.items():
        stats_table.add_row(k.replace("_", " ").title(), str(v))

    console.print(stats_table)

    # Display Visual Elements Extracted
    if doc.visual_elements:
        elem_table = Table(title="Extracted & Annotated Visual Assets", show_header=True, header_style="bold magenta")
        elem_table.add_column("Element ID", width=22)
        elem_table.add_column("Type", width=10)
        elem_table.add_column("Page", width=6)
        elem_table.add_column("Bounding Box", width=26)
        elem_table.add_column("Description Preview", width=45)

        for elem in doc.visual_elements[:10]:  # Top 10 preview
            bbox_str = f"[{elem.bounding_box.x0:.0f}, {elem.bounding_box.y0:.0f}, {elem.bounding_box.x1:.0f}, {elem.bounding_box.y1:.0f}]"
            desc_preview = (elem.description[:60] + "...") if len(elem.description) > 60 else elem.description
            elem_table.add_row(
                elem.element_id,
                elem.element_type.value,
                str(elem.page_number),
                bbox_str,
                desc_preview.replace("\n", " "),
            )

        console.print(elem_table)

    # Display Sample Chunks
    sample_table = Table(title="Sample Injected Retrieval Chunks", show_header=True, header_style="bold cyan")
    sample_table.add_column("Chunk ID", width=24)
    sample_table.add_column("Type", width=12)
    sample_table.add_column("Page", width=6)
    sample_table.add_column("Content Preview", width=50)

    for chunk in doc.chunks[:5]:
        preview = (chunk.content[:80] + "...") if len(chunk.content) > 80 else chunk.content
        sample_table.add_row(
            chunk.chunk_id,
            chunk.chunk_type.value,
            str(chunk.page_number),
            preview.replace("\n", " "),
        )

    console.print(sample_table)
    console.print(f"[bold green]✔ Done![/bold green] Ingestion artifact saved to data/processed/{doc.document_id}.json")


if __name__ == "__main__":
    main()
