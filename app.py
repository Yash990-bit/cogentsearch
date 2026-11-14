"""
CogentSearch — Multimodal Agentic RAG Streamlit Application (Phase 4).
Features:
1. Sidebar: File uploader (PDF, DOCX, PPTX), document ingestion pipeline, and collection stats.
2. Main Window: Chat interface with LangGraph agent loop (Rewriter -> Hybrid Retriever -> Grader -> Generator).
3. Exact Page Citations: Displays formatted badges and source page links.
4. Multimodal References: Displays cropped tables and images with bounding box provenance.
"""

from __future__ import annotations
import os
import io
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="CogentSearch — Multimodal Agentic RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern enterprise aesthetic
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .citation-badge {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 4px 4px 4px 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .citation-badge-table {
        background-color: #1e1b4b;
        color: #a78bfa;
        border: 1px solid #7c3aed;
    }
    .citation-badge-image {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
    }
    .trace-card {
        background-color: #111827;
        border-left: 3px solid #38bdf8;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 0.88rem;
    }
    .stat-box {
        background-color: #1f2937;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 1px solid #374151;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Imports from CogentSearch core
from cogent.core.config import settings
from cogent.schemas.indexing import IndexedDocument, SearchResultChunk
from cogent.schemas.document import BoundingBox
from cogent.ingestion.pipeline import MultimodalIngestionPipeline
from cogent.vectorstore.splitter import SemanticDocumentSplitter
from cogent.vectorstore.qdrant_store import QdrantVectorStore
from cogent.retrieval.bm25_retriever import BM25Retriever
from cogent.retrieval.hybrid_retriever import HybridRetriever
from cogent.agent.agentic_loop import AgenticRAG, AgenticResponse

# Set up local directories
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR = Path("data/test_corpus")


def get_bounding_box_crop(
    file_path: str | Path,
    page_number: int,
    bbox: Optional[Dict[str, Any] | BoundingBox],
) -> Optional[bytes]:
    """Generates high-resolution PNG image crop of a bounding box from a PDF page using PyMuPDF."""
    path = Path(file_path)
    if not path.exists():
        # Check in test corpus or uploads
        candidate1 = UPLOAD_DIR / path.name
        candidate2 = CORPUS_DIR / path.name
        if candidate1.exists():
            path = candidate1
        elif candidate2.exists():
            path = candidate2
        else:
            return None

    if not path.suffix.lower() == ".pdf":
        return None

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        if page_number < 1 or page_number > len(doc):
            return None
        page = doc[page_number - 1]

        if bbox is None:
            # Render top half of page as preview
            rect = fitz.Rect(0, 0, page.rect.width, page.rect.height / 2)
        elif isinstance(bbox, dict):
            rect = fitz.Rect(bbox.get("x0", 0), bbox.get("y0", 0), bbox.get("x1", 500), bbox.get("y1", 400))
        elif isinstance(bbox, BoundingBox):
            rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        else:
            rect = page.rect

        pix = page.get_pixmap(clip=rect, dpi=160)
        return pix.tobytes("png")
    except Exception:
        return None


def init_session_state():
    """Initializes persistent pipeline and session state objects."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "hybrid_retriever" not in st.session_state:
        # Initialize in-memory persistent stores
        qdrant = QdrantVectorStore(in_memory=True, collection_name="col_streamlit_ui")
        bm25 = BM25Retriever()
        st.session_state.hybrid_retriever = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)

    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()

    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs = []


init_session_state()
retriever: HybridRetriever = st.session_state.hybrid_retriever
agent = AgenticRAG(hybrid_retriever=retriever, max_retries=2)


# ==============================================================================
# SIDEBAR: DOCUMENT INGESTION & STATS
# ==============================================================================
with st.sidebar:
    st.title("⚡ CogentSearch")
    st.caption("Phase 4: Multimodal Agentic RAG System")

    st.markdown("---")
    st.subheader("📂 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload Multimodal Documents",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        help="Upload enterprise PDFs, Word documents, or PowerPoint decks.",
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            with st.status(f"Ingesting {len(new_files)} document(s)...", expanded=True) as status:
                pipeline = MultimodalIngestionPipeline()
                splitter = SemanticDocumentSplitter()

                for uploaded_file in new_files:
                    st.write(f"Parsing `{uploaded_file.name}`...")
                    dest_path = UPLOAD_DIR / uploaded_file.name
                    with open(dest_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        multimodal_doc = pipeline.process_file(dest_path)
                        chunks = splitter.split_document(multimodal_doc)
                        retriever.index_documents(chunks)
                        st.session_state.indexed_docs.extend(chunks)
                        st.session_state.indexed_files.add(uploaded_file.name)
                        st.write(f"✔ Indexed {len(chunks)} chunks from `{uploaded_file.name}`.")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

                status.update(label="Ingestion and hybrid indexing complete!", state="complete", expanded=False)

    # Pre-load sample test corpus button
    if CORPUS_DIR.exists():
        sample_pdfs = list(CORPUS_DIR.glob("*.pdf"))[:3]
        if st.button("📥 Load Sample Enterprise Corpus", use_container_width=True):
            with st.status("Loading sample benchmark documents...", expanded=True) as status:
                pipeline = MultimodalIngestionPipeline()
                splitter = SemanticDocumentSplitter()
                for p in sample_pdfs:
                    if p.name not in st.session_state.indexed_files:
                        st.write(f"Indexing `{p.name}`...")
                        try:
                            mdoc = pipeline.process_file(p)
                            chunks = splitter.split_document(mdoc)
                            retriever.index_documents(chunks)
                            st.session_state.indexed_docs.extend(chunks)
                            st.session_state.indexed_files.add(p.name)
                        except Exception as e:
                            st.warning(f"Could not load {p.name}: {e}")
                status.update(label="Sample corpus loaded!", state="complete", expanded=False)
            st.rerun()

    st.markdown("---")
    st.subheader("📊 Collection Analytics")
    total_chunks = len(st.session_state.indexed_docs)
    table_chunks = sum(1 for c in st.session_state.indexed_docs if c.modality_type == "table")
    image_chunks = sum(1 for c in st.session_state.indexed_docs if c.modality_type == "image")
    text_chunks = total_chunks - table_chunks - image_chunks

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Chunks", total_chunks)
        st.metric("Tables", table_chunks)
    with col2:
        st.metric("Documents", len(st.session_state.indexed_files))
        st.metric("Images/Charts", image_chunks)

    st.markdown("---")
    st.subheader("⚙️ System Configuration")
    st.markdown(
        f"""
        - **Embedding Engine**: `{settings.EMBEDDING_PROVIDER}` ({settings.EMBEDDING_DIMENSION}d Cosine)
        - **Lexical Engine**: Persistent BM25 (Okapi)
        - **Reranker**: `{settings.CROSS_ENCODER_MODEL.split('/')[-1]}`
        - **Rewriter/Grader LLM**: `{settings.REWRITER_MODEL_NAME}`
        - **Generator LLM**: `{settings.GENERATOR_MODEL_NAME}`
        """
    )

    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        st.session_state.messages = []
        st.session_state.indexed_files = set()
        st.session_state.indexed_docs = []
        qdrant = QdrantVectorStore(in_memory=True, collection_name="col_streamlit_ui_reset")
        bm25 = BM25Retriever()
        st.session_state.hybrid_retriever = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
        st.success("Knowledge base reset!")
        st.rerun()


# ==============================================================================
# MAIN CHAT WINDOW
# ==============================================================================
st.title("⚡ CogentSearch: Multimodal Agentic Assistant")
st.markdown(
    "Query across unstructured enterprise PDF reports, complex multipage tables, and architectural slides "
    "with **strict spatial provenance grounding** and self-correcting agent execution."
)

# Suggested queries
if not st.session_state.messages:
    st.markdown("##### 💡 Suggested Questions:")
    sample_queries = [
        "What data is in the multipage financial table?",
        "What parameters and expenditure thresholds are specified in protocol FIN-10?",
        "What is the peak continuous electrical load for subsystem SYS-400?",
        "What microscopic failure mode do turbine blades exhibit under thermal cycling?",
    ]
    cols = st.columns(len(sample_queries))
    for i, sq in enumerate(sample_queries):
        if cols[i].button(sq, key=f"sq_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": sq})
            st.rerun()

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Display saved citations and references
        if "citations" in msg and msg["citations"]:
            st.markdown("##### 📌 Verified Evidence Citations:")
            citation_html = []
            for c in msg["citations"]:
                mod = c.get("modality_type", "Text").lower()
                badge_class = "citation-badge"
                if "table" in mod:
                    badge_class += " citation-badge-table"
                elif "image" in mod or "visual" in mod:
                    badge_class += " citation-badge-image"

                tag = c.get("citation_tag") or f"[Source: {c.get('source_file')}, Page {c.get('page_number')}, Type: {c.get('modality_type')}]"
                citation_html.append(f"<span class='{badge_class}'>{tag}</span>")
            st.markdown(" ".join(citation_html), unsafe_allow_html=True)

        if "crops" in msg and msg["crops"]:
            st.markdown("##### 🖼️ Multimodal References (Crops & Visual Tables):")
            crop_cols = st.columns(min(3, len(msg["crops"])))
            for idx, crop_item in enumerate(msg["crops"]):
                col_target = crop_cols[idx % len(crop_cols)]
                with col_target:
                    if crop_item.get("image_bytes"):
                        st.image(crop_item["image_bytes"], caption=crop_item.get("caption", "Visual Evidence"), use_container_width=True)
                    elif crop_item.get("table_markdown"):
                        st.markdown(f"**Table Reference ({crop_item.get('source')}, P.{crop_item.get('page')}):**")
                        st.markdown(crop_item["table_markdown"])

        if "trace" in msg and msg["trace"]:
            with st.expander("🔍 Agent Reasoning Trace"):
                for step in msg["trace"]:
                    st.markdown(f"- `{step}`")


# User input handling
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Executing CogentSearch Agentic Loop (Rewriter -> Hybrid Retriever -> Grader -> Generator)..."):
            resp: AgenticResponse = agent.run(prompt)

            # Display synthesized generation
            st.markdown(resp.generation)

            # Display Agent Execution Trace
            with st.expander("🔍 Agent Reasoning & Verification Trace", expanded=False):
                st.markdown(f"**Optimized Search Query:** `{resp.rewritten_query}`")
                st.markdown(f"**Retrieved Chunks Count:** `{resp.retrieved_chunk_count}`")
                st.markdown(f"**Web Search Triggered:** `{resp.web_search_used}`")
                st.markdown(f"**Rewrite Attempts:** `{resp.retry_count}`")
                st.markdown("**Graph Traversal:**")
                for t in resp.trace:
                    st.markdown(f"<div class='trace-card'>• {t}</div>", unsafe_allow_html=True)

            # Display Citations
            formatted_citations = []
            if resp.citations:
                st.markdown("##### 📌 Verified Citations:")
                badges = []
                for cit in resp.citations:
                    mod = cit.modality_type.lower()
                    badge_class = "citation-badge"
                    if "table" in mod:
                        badge_class += " citation-badge-table"
                    elif "image" in mod or "visual" in mod:
                        badge_class += " citation-badge-image"
                    badges.append(f"<span class='{badge_class}'>{cit.citation_tag}</span>")
                    formatted_citations.append(cit.model_dump())
                st.markdown(" ".join(badges), unsafe_allow_html=True)

            # Extract Cropped Tables / Images
            visual_crops = []
            if resp.retrieved_documents:
                for chunk in resp.retrieved_documents:
                    # Table references
                    if chunk.modality_type == "table":
                        visual_crops.append({
                            "type": "table",
                            "source": chunk.source_file,
                            "page": chunk.page_number,
                            "table_markdown": chunk.content,
                            "caption": f"Table: {chunk.source_file} (Page {chunk.page_number})",
                        })
                    # Image references with BoundingBox crops
                    elif chunk.modality_type == "image":
                        crop_bytes = get_bounding_box_crop(chunk.source_file, chunk.page_number, chunk.bounding_box)
                        if crop_bytes:
                            visual_crops.append({
                                "type": "image",
                                "source": chunk.source_file,
                                "page": chunk.page_number,
                                "image_bytes": crop_bytes,
                                "caption": f"Image Asset: {chunk.source_file} (Page {chunk.page_number})",
                            })

            # Display cropped tables and images
            if visual_crops:
                st.markdown("##### 🖼️ Grounded Visual Elements & Tables:")
                crop_cols = st.columns(min(3, len(visual_crops)))
                for idx, v_item in enumerate(visual_crops[:6]):
                    with crop_cols[idx % len(crop_cols)]:
                        if v_item.get("image_bytes"):
                            st.image(v_item["image_bytes"], caption=v_item["caption"], use_container_width=True)
                        elif v_item.get("table_markdown"):
                            st.markdown(f"**{v_item['caption']}**")
                            st.markdown(v_item["table_markdown"])

            # Save assistant message to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.generation,
                "citations": formatted_citations,
                "crops": visual_crops,
                "trace": resp.trace,
            })
