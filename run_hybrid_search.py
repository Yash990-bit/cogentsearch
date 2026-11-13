"""
CLI Runner for CogentSearch Phase 2: Hybrid Indexing & Retrieval.
Usage:
    python run_hybrid_search.py --query "What is the status of TX-1001?"
    python run_hybrid_search.py --query "Show me the quarterly operating margin chart"
    python run_hybrid_search.py --index-all
"""

import sys
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from cogent.schemas.document import MultimodalDocument
from cogent.schemas.indexing import IndexedDocument, SearchResultChunk
from cogent.vectorstore.splitter import SemanticDocumentSplitter
from cogent.vectorstore.qdrant_store import QdrantVectorStore
from cogent.retrieval.bm25_retriever import BM25Retriever
from cogent.retrieval.cross_encoder import CrossEncoderReranker
from cogent.retrieval.hybrid_retriever import HybridRetriever


console = Console()


def load_all_processed_documents() -> list[IndexedDocument]:
    """Loads all processed documents from data/processed and splits into IndexedDocuments."""
    splitter = SemanticDocumentSplitter(chunk_size=800, chunk_overlap=150)
    all_indexed: list[IndexedDocument] = []

    proc_dir = Path("data/processed")
    if not proc_dir.exists():
        console.print("[red]No processed documents found in data/processed! Run ingestion first.[/red]")
        return []

    for jf in proc_dir.glob("*.json"):
        with open(jf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                doc = MultimodalDocument(**data)
                chunks = splitter.split_document(doc)
                all_indexed.extend(chunks)
            except Exception as e:
                console.print(f"[yellow]Skipping {jf.name}: {e}[/yellow]")

    return all_indexed


def main():
    parser = argparse.ArgumentParser(description="CogentSearch Phase 2: Hybrid Search & Cross-Encoder Reranking")
    parser.add_argument("--query", "-q", type=str, default="What is the transaction amount and status for TX-1001?", help="Search query")
    parser.add_argument("--modality", "-m", type=str, choices=["text", "table", "image"], default=None, help="Filter by modality")
    parser.add_argument("--top-k", "-k", type=int, default=4, help="Final top-k chunks to return after Cross-Encoder reranking")

    args = parser.parse_args()

    console.print(
        Panel(
            f"[bold cyan]CogentSearch Hybrid Retrieval & Cross-Encoder[/bold cyan]\n"
            f"Query: [yellow]{args.query}[/yellow]\n"
            f"Filter Modality: [magenta]{args.modality or 'All (text, table, image)'}[/magenta]\n"
            f"Pipeline: [green]Dense (Qdrant) + Sparse (BM25) ➜ RRF (Top 10) ➜ Cross-Encoder ➜ Top {args.top_k}[/green]",
            title="[bold blue]Phase 2: Hybrid Indexing & Storage[/bold blue]",
            expand=False,
        )
    )

    # 1. Load documents
    with console.status("[bold green]Loading and chunking documents from data/processed..."):
        all_chunks = load_all_processed_documents()

    if not all_chunks:
        console.print("[bold red]Error:[/bold red] No indexed chunks available.")
        sys.exit(1)

    console.print(f"Loaded [bold green]{len(all_chunks)}[/bold green] total indexed chunks.")

    # 2. Initialize Qdrant and BM25
    with console.status("[bold green]Indexing into local Qdrant and BM25..."):
        qdrant = QdrantVectorStore(in_memory=True)
        bm25 = BM25Retriever()
        cross_encoder = CrossEncoderReranker(use_local_fallback=False)
        hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25, cross_encoder=cross_encoder)
        hybrid.index_documents(all_chunks)

    # 3. Perform Hybrid Search & Reranking
    with console.status("[bold green]Executing Hybrid Retrieval & Cross-Encoder Reranking..."):
        results = hybrid.search(
            query=args.query,
            top_k_dense=20,
            top_k_bm25=20,
            top_n_rerank=10,
            final_top_k=args.top_k,
            filter_modality=args.modality,
        )

    # 4. Render Results
    if not results:
        console.print("[yellow]No relevant chunks matched the query.[/yellow]")
        return

    table = Table(
        title=f"Top {len(results)} Most Relevant Chunks (Cross-Encoder Reranked)",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Rank", width=5, justify="center")
    table.add_column("Modality", width=10, justify="center")
    table.add_column("Page", width=6, justify="center")
    table.add_column("Source File", width=25)
    table.add_column("CE Score", width=10, justify="right")
    table.add_column("RRF Score", width=10, justify="right")
    table.add_column("Content Snippet", width=45)

    for res in results:
        mod_style = "cyan" if res.modality_type == "text" else ("magenta" if res.modality_type == "table" else "yellow")
        snippet = (res.content[:90] + "...") if len(res.content) > 90 else res.content
        table.add_row(
            f"#{res.rank}",
            f"[{mod_style}]{res.modality_type.upper()}[/{mod_style}]",
            str(res.page_number),
            res.source_file,
            f"{res.cross_encoder_score:.4f}" if res.cross_encoder_score is not None else "N/A",
            f"{res.rrf_score:.5f}" if res.rrf_score is not None else "N/A",
            snippet.replace("\n", " "),
        )

    console.print(table)


if __name__ == "__main__":
    main()

