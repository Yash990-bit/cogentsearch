"""
CogentSearch — Enterprise Multimodal Agentic Document Intelligence (Chainlit UI).

Phase 4 Enterprise UI:
1. Drag-and-Drop Ingestion: Accepts PDF, PPTX, and DOCX with live ingestion progress.
2. Full Chain of Thought (CoT): Live LangGraph state machine visualization using cl.step:
   "Rewriting Query" ➔ "Searching Vector DB" ➔ "Grading Evidence" ➔ "Drafting Answer"
3. Multimodal Provenance & Citations:
   - Cropped high-resolution bounding-box images (cl.Image) for tables and figures.
   - Clickable side-sheet drawers (cl.Text) for full text citations and page references.
"""

from __future__ import annotations

import os
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

import chainlit as cl
from loguru import logger

# CogentSearch core imports
from cogent.core.config import settings
from cogent.schemas.indexing import IndexedDocument, SearchResultChunk
from cogent.schemas.document import BoundingBox, MultimodalDocument
from cogent.ingestion.pipeline import MultimodalIngestionPipeline
from cogent.vectorstore.splitter import SemanticDocumentSplitter
from cogent.vectorstore.qdrant_store import QdrantVectorStore
from cogent.retrieval.bm25_retriever import BM25Retriever
from cogent.retrieval.hybrid_retriever import HybridRetriever
from cogent.agent.agentic_loop import AgenticRAG
from cogent.agent.graph import build_agentic_rag_graph
from cogent.agent.state import AgentState, CitationItem

# Setup workspace directories
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR = Path("data/test_corpus")


def get_bounding_box_crop(
    file_path: str | Path,
    page_number: int,
    bbox: Optional[Dict[str, Any] | BoundingBox] = None,
) -> Optional[bytes]:
    """
    Extracts a high-resolution PNG image crop from a document page using PyMuPDF.
    Resolves file paths across uploads, corpus directory, and absolute references.
    """
    path = Path(file_path)
    if not path.exists():
        for candidate in [UPLOAD_DIR / path.name, CORPUS_DIR / path.name]:
            if candidate.exists():
                path = candidate
                break
        else:
            return None

    if path.suffix.lower() != ".pdf":
        return None

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        if page_number < 1 or page_number > len(doc):
            return None
        page = doc[page_number - 1]

        if bbox is None:
            # Crop top portion of the page as preview
            rect = fitz.Rect(0, 0, page.rect.width, min(page.rect.height / 2, 450))
        elif isinstance(bbox, dict):
            rect = fitz.Rect(
                bbox.get("x0", 0),
                bbox.get("y0", 0),
                bbox.get("x1", page.rect.width),
                bbox.get("y1", page.rect.height / 2),
            )
        elif hasattr(bbox, "x0"):
            rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        else:
            rect = page.rect

        pix = page.get_pixmap(clip=rect, dpi=160)
        return pix.tobytes("png")
    except Exception as exc:
        logger.warning(f"Error extracting bounding box crop from {path}: {exc}")
        return None


def ingest_document_file(
    file_path: Path,
    pipeline: MultimodalIngestionPipeline,
    splitter: SemanticDocumentSplitter,
    retriever: HybridRetriever,
) -> tuple[MultimodalDocument, List[IndexedDocument]]:
    """Ingests a single document through Phase 1 parsing and Phase 2 indexing."""
    multimodal_doc = pipeline.process_file(file_path)
    indexed_chunks = splitter.split_document(multimodal_doc)

    if indexed_chunks:
        retriever.qdrant_store.upsert_documents(indexed_chunks)
        retriever.bm25_retriever.index_documents(indexed_chunks)

    return multimodal_doc, indexed_chunks


@cl.step(name="🔍 Rewriting Query", type="tool")
async def step_rewrite_query(question: str, rewritten_query: str, attempt: int = 1) -> str:
    """Chainlit step showing Query Rewriter progress and output."""
    output = f"**Raw User Query:**\n> {question}\n\n**Optimized Vector Search Query (Attempt {attempt}):**\n`{rewritten_query}`"
    return output


def format_vector_search_step(query: str, documents: List[SearchResultChunk]) -> str:
    """Formats Phase 2 Hybrid Retrieval results and neural reranking output."""
    lines = [
        f"**Search Query:** `{query}`\n",
        f"**Retrieved & Reranked:** `{len(documents)} chunks`\n",
    ]
    for i, doc in enumerate(documents[:5]):
        modality = getattr(doc, "modality_type", "text").upper()
        rrf = getattr(doc, "rrf_score", None)
        rrf_str = f"{rrf:.4f}" if isinstance(rrf, (int, float)) else "N/A"
        ce = getattr(doc, "cross_encoder_score", None) or getattr(doc, "score", None)
        ce_str = f"{ce:.4f}" if isinstance(ce, (int, float)) else "N/A"
        lines.append(
            f"{i+1}. **[{modality}]** `{doc.source_file}` (p. {doc.page_number}) — "
            f"RRF: `{rrf_str}` | Cross-Encoder: `{ce_str}`"
        )
    return "\n".join(lines)


@cl.step(name="⚡ Searching Vector DB (Hybrid Dense + BM25)", type="tool")
async def step_vector_search(query: str, documents: List[SearchResultChunk]) -> str:
    """Chainlit step showing Phase 2 Hybrid Retrieval results and neural reranking."""
    return format_vector_search_step(query, documents)


@cl.step(name="⚖️ Grading Evidence (Self-Correction)", type="tool")
async def step_grade_evidence(is_relevant: bool, reason: str) -> str:
    """Chainlit step showing LLM Document Grader evaluation."""
    verdict = "✅ **SUFFICIENT** — Answer present in retrieved passages" if is_relevant else "⚠️ **INSUFFICIENT** — Self-correction loop triggered"
    return f"**Verdict:** {verdict}\n\n**Evaluator Rationale:**\n{reason}"


@cl.step(name="🌐 Web Search Fallback", type="tool")
async def step_web_search(query: str, results: List[Any]) -> str:
    """Chainlit step showing DuckDuckGo web search fallback."""
    return f"Retrieved **{len(results)} external web references** via live search for: `{query}`"


@cl.step(name="📝 Drafting Answer (Strict Citations)", type="tool")
async def step_draft_answer(citations_count: int) -> str:
    """Chainlit step showing answer generation and Pydantic schema validation."""
    return f"Generated synthesized response with **{citations_count} verified citations** adhering to Pydantic structured schemas."


@cl.on_chat_start
async def on_chat_start():
    """Initializes the Chainlit session with persistent in-memory Qdrant and BM25 retrievers."""
    logger.info("Initializing CogentSearch session state...")

    # Initialize Phase 2 Hybrid Retriever (isolated in-memory collection per session)
    session_id = cl.user_session.get("id", "default")
    collection_name = f"col_chainlit_{session_id[:8]}"
    qdrant = QdrantVectorStore(in_memory=True, collection_name=collection_name)
    bm25 = BM25Retriever()
    hybrid_retriever = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)

    # Initialize LangGraph Agentic RAG
    compiled_graph = build_agentic_rag_graph(hybrid_retriever=hybrid_retriever, max_retries=2)

    # Store in Chainlit session
    cl.user_session.set("hybrid_retriever", hybrid_retriever)
    cl.user_session.set("compiled_graph", compiled_graph)
    cl.user_session.set("indexed_files", set())
    cl.user_session.set("total_chunks", 0)

    # Welcome message with action button
    welcome_content = (
        "# ⚡ CogentSearch Enterprise Intelligence\n"
        "**Multimodal Agentic Document Intelligence System (Phase 1–4)**\n\n"
        "Welcome! This system integrates **LlamaParse multimodal ingestion**, "
        "**hybrid dense + BM25 retrieval**, a **self-correcting LangGraph agent loop**, "
        "and **strict line-of-sight citations** with cropped tables & charts.\n\n"
        "### How to get started:\n"
        "1. **Drag and drop** PDF, PPTX, or DOCX files into the chat to ingest them.\n"
        "2. Click the button below to **instantly preload the multimodal sample corpus**.\n"
        "3. Ask any technical, financial, or visual document question!"
    )

    actions = [
        cl.Action(
            name="load_sample_corpus",
            payload={"action": "preload"},
            label="📂 Preload Sample Corpus (Financial & Multimodal)",
            tooltip="Indexes multi-page tables, charts, slides, and reports from data/test_corpus",
        )
    ]

    await cl.Message(content=welcome_content, actions=actions).send()


@cl.action_callback("load_sample_corpus")
async def on_load_sample_corpus(action: cl.Action):
    """Callback to load and index the test corpus in one click."""
    retriever: HybridRetriever = cl.user_session.get("hybrid_retriever")
    indexed_files: set = cl.user_session.get("indexed_files", set())

    sample_files = [
        CORPUS_DIR / "doc_02_multipage_table.pdf",
        CORPUS_DIR / "doc_04_charts_diagrams.pdf",
        CORPUS_DIR / "doc_05_nested_report.docx",
        CORPUS_DIR / "doc_06_slides_20plus.pptx",
    ]

    available_files = [f for f in sample_files if f.exists()]
    if not available_files:
        await cl.Message(content="⚠️ No sample files found in `data/test_corpus`.").send()
        return

    async with cl.Step(name="Ingesting and chunking documents...", type="tool") as step:
        step.input = f"Ingesting {len(available_files)} sample files: {', '.join(f.name for f in available_files)}"
        
        pipeline = MultimodalIngestionPipeline()
        splitter = SemanticDocumentSplitter()
        new_chunks_count = 0

        for f in available_files:
            if f.name in indexed_files:
                continue
            doc, chunks = await asyncio.to_thread(ingest_document_file, f, pipeline, splitter, retriever)
            indexed_files.add(f.name)
            new_chunks_count += len(chunks)

        cl.user_session.set("indexed_files", indexed_files)
        current_total = cl.user_session.get("total_chunks", 0) + new_chunks_count
        cl.user_session.set("total_chunks", current_total)

        step.output = (
            f"✅ **Ingestion Complete!**\n\n"
            f"- **Files Indexed:** `{len(indexed_files)} files`\n"
            f"- **Total Chunks in Qdrant & BM25:** `{current_total} chunks`\n"
            f"- **Modalities:** Text, Markdown Tables, High-Res Chart Crops"
        )

    await cl.Message(
        content=f"🎉 **Preloaded {len(available_files)} documents into Hybrid Store!**\n\n"
                f"You can now ask questions like:\n"
                f"- *'What are the key financial metrics in the quarterly report table?'*\n"
                f"- *'Summarize the system architecture slides.'*\n"
                f"- *'What are the latency benchmarks in the chart?'*"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Main message handler:
    1. Handles drag-and-drop file attachments (PDF, DOCX, PPTX) with live ingestion status.
    2. Runs the LangGraph StateGraph agent loop with live Chain of Thought dropdowns.
    3. Emits multimodal citations with inline cropped tables/images and side-sheet text quotes.
    """
    retriever: HybridRetriever = cl.user_session.get("hybrid_retriever")
    compiled_graph = cl.user_session.get("compiled_graph")
    indexed_files: set = cl.user_session.get("indexed_files", set())

    # --------------------------------------------------------------------------
    # 1. FILE MANAGEMENT: INGEST DRAG-AND-DROP ATTACHMENTS
    # --------------------------------------------------------------------------
    files_to_process = [
        el for el in message.elements
        if getattr(el, "path", None) and Path(el.path).is_file()
    ]

    if files_to_process:
        async with cl.Step(name="Ingesting and chunking documents...", type="tool") as step:
            step.input = f"Received {len(files_to_process)} uploaded file(s): {', '.join(el.name for el in files_to_process)}"
            pipeline = MultimodalIngestionPipeline()
            splitter = SemanticDocumentSplitter()
            added_chunks = 0

            for el in files_to_process:
                src_path = Path(el.path)
                dest_path = UPLOAD_DIR / el.name
                shutil.copy2(src_path, dest_path)

                doc, chunks = await asyncio.to_thread(
                    ingest_document_file, dest_path, pipeline, splitter, retriever
                )
                indexed_files.add(el.name)
                added_chunks += len(chunks)

            cl.user_session.set("indexed_files", indexed_files)
            total = cl.user_session.get("total_chunks", 0) + added_chunks
            cl.user_session.set("total_chunks", total)

            step.output = (
                f"✅ **Processed {len(files_to_process)} document(s)!**\n"
                f"Added **{added_chunks} new multimodal chunks** into hybrid vector & BM25 store."
            )

        if not message.content.strip():
            await cl.Message(
                content=f"✅ **Uploaded & Indexed:** {', '.join(el.name for el in files_to_process)}.\n\nAsk any question regarding these documents!"
            ).send()
            return

    # If no query text provided
    user_query = message.content.strip()
    if not user_query:
        return

    # --------------------------------------------------------------------------
    # 2. "CHAIN OF THOUGHT" VISIBILITY: STREAM LANGGRAPH AGENTIC LOOP
    # --------------------------------------------------------------------------
    initial_state: AgentState = {
        "question": user_query,
        "rewritten_query": "",
        "documents": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_relevant": False,
        "grade_reason": "",
        "web_search_needed": False,
        "web_documents": [],
        "generation": "",
        "citations": [],
        "trace": [],
    }

    current_state = dict(initial_state)

    # Stream LangGraph state machine node-by-node
    async for chunk in compiled_graph.astream(initial_state):
        for node_name, node_update in chunk.items():
            current_state.update(node_update)

            try:
                if node_name == "query_rewriter":
                    attempt = current_state.get("retry_count", 0) + 1
                    await step_rewrite_query(
                        question=current_state["question"],
                        rewritten_query=node_update.get("rewritten_query", ""),
                        attempt=attempt,
                    )

                elif node_name == "retriever":
                    await step_vector_search(
                        query=current_state.get("rewritten_query", current_state["question"]),
                        documents=node_update.get("documents", []),
                    )

                elif node_name == "doc_grader":
                    await step_grade_evidence(
                        is_relevant=node_update.get("is_relevant", False),
                        reason=node_update.get("grade_reason", "No evaluation details."),
                    )

                elif node_name == "web_search":
                    await step_web_search(
                        query=current_state.get("rewritten_query", current_state["question"]),
                        results=node_update.get("web_documents", []),
                    )

                elif node_name == "generator":
                    citations = node_update.get("citations", [])
                    await step_draft_answer(citations_count=len(citations))
            except Exception as step_err:
                logger.warning(f"Error rendering step for node {node_name}: {step_err}")

    # --------------------------------------------------------------------------
    # 3. MULTIMODAL CITATIONS & PROVENANCE CROPS
    # --------------------------------------------------------------------------
    final_generation = current_state.get("generation", "No answer generated.")
    raw_citations = current_state.get("citations", [])
    documents: List[SearchResultChunk] = current_state.get("documents", [])

    elements = []
    seen_crops = set()

    # Process multimodal elements (tables and figures)
    for cit in raw_citations:
        src_file = cit.get("source_file", "")
        page_num = cit.get("page_number", 1)
        modality = cit.get("modality_type", "text").lower()
        bbox = cit.get("bounding_box")

        # Table & Chart visual bounding-box crop
        if modality in ["table", "image", "figure"]:
            crop_key = f"{src_file}_{page_num}_{modality}"
            if crop_key not in seen_crops:
                seen_crops.add(crop_key)
                crop_bytes = get_bounding_box_crop(
                    file_path=src_file,
                    page_number=page_num,
                    bbox=bbox,
                )
                if crop_bytes:
                    elements.append(
                        cl.Image(
                            name=f"Cropped {modality.title()} (Page {page_num}) — {src_file}",
                            content=crop_bytes,
                            display="inline",
                        )
                    )

        # Verified text passage citation for side drawer
        matched_text = cit.get("matched_chunk_text", "")
        if not matched_text:
            # Fallback to matching document text
            for d in documents:
                if d.source_file == src_file and d.page_number == page_num:
                    matched_text = getattr(d, "content", "") or getattr(d, "text", "")
                    break

        element_name = f"Source [{src_file}, Page {page_num}]"
        evidence_content = (
            f"### Document Evidence: {src_file}\n"
            f"- **Page:** {page_num}\n"
            f"- **Modality:** `{modality.upper()}`\n\n"
            f"#### Extracted Verified Chunk:\n"
            f"```text\n{matched_text.strip()}\n```"
        )
        elements.append(
            cl.Text(
                name=element_name,
                content=evidence_content,
                display="side",
            )
        )

    # Send final response with integrated multimodal elements
    await cl.Message(
        content=final_generation,
        elements=elements,
    ).send()
