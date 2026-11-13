"""
Integration and Unit Tests for CogentSearch Phase 2: Hybrid Indexing & Storage.
Verifies Document Schema, Semantic Splitting, Qdrant Vector Store, BM25 Indexing,
Reciprocal Rank Fusion (RRF), and Cross-Encoder Re-Ranking.
"""

import pytest
from pathlib import Path
from cogent.schemas.document import BoundingBox, ChunkType
from cogent.schemas.indexing import IndexedDocument, SearchResultChunk
from cogent.vectorstore.splitter import SemanticDocumentSplitter
from cogent.vectorstore.embeddings import EmbeddingEngine
from cogent.vectorstore.qdrant_store import QdrantVectorStore
from cogent.retrieval.bm25_retriever import BM25Retriever
from cogent.retrieval.cross_encoder import CrossEncoderReranker
from cogent.retrieval.hybrid_retriever import HybridRetriever
from cogent.ingestion.pipeline import MultimodalIngestionPipeline


@pytest.fixture
def sample_indexed_documents():
    """Generates synthetic multi-modal indexed documents for testing retrieval."""
    bbox1 = BoundingBox(page_number=1, x0=50.0, y0=100.0, x1=500.0, y1=250.0, page_width=595.0, page_height=842.0)
    bbox2 = BoundingBox(page_number=2, x0=60.0, y0=120.0, x1=400.0, y1=380.0, page_width=595.0, page_height=842.0)

    docs = [
        IndexedDocument(
            chunk_id="doc1_chunk_0",
            content="Chapter 1: System Overview. CogentSearch is an agentic multimodal retrieval architecture.",
            modality_type="text",
            page_number=1,
            bounding_box=None,
            source_file="architecture.pdf",
            headers_hierarchy=["System Overview"],
        ),
        IndexedDocument(
            chunk_id="doc1_chunk_1",
            content="| Transaction ID | Amount ($) | Status |\n| --- | --- | --- |\n| TX-1001 | $145,200.00 | Settled |\n| TX-1002 | $89,500.00 | Pending |",
            modality_type="table",
            page_number=1,
            bounding_box=bbox1,
            source_file="financial_ledger.pdf",
            headers_hierarchy=["Ledger", "Transactions"],
        ),
        IndexedDocument(
            chunk_id="doc1_chunk_2",
            content="### [Visual Asset on Page 2: IMAGE]\n**Description:** Quarterly operating margin performance chart displaying 28% growth across Q1 to Q4.",
            modality_type="image",
            page_number=2,
            bounding_box=bbox2,
            source_file="financial_ledger.pdf",
            headers_hierarchy=["Financial Visuals", "Operating Margin"],
        ),
        IndexedDocument(
            chunk_id="doc1_chunk_3",
            content="Section 3: Kubernetes deployment guidelines for distributed worker clusters and quota limits.",
            modality_type="text",
            page_number=3,
            bounding_box=None,
            source_file="deployment_guide.pdf",
            headers_hierarchy=["Deployment", "Kubernetes"],
        ),
        IndexedDocument(
            chunk_id="doc1_chunk_4",
            content="Transaction reconciliation procedure for audit logs and international wire settlements.",
            modality_type="text",
            page_number=4,
            bounding_box=None,
            source_file="financial_ledger.pdf",
            headers_hierarchy=["Reconciliation"],
        ),
    ]
    return docs


def test_indexed_document_schema():
    """Validates that IndexedDocument strictly requires content, modality_type, page_number, bounding_box, source_file."""
    bbox = BoundingBox(page_number=1, x0=10.0, y0=20.0, x1=100.0, y1=100.0)
    doc = IndexedDocument(
        chunk_id="test_id_1",
        content="Sample textual content",
        modality_type="text",
        page_number=1,
        bounding_box=bbox,
        source_file="test_doc.pdf",
    )
    assert doc.content == "Sample textual content"
    assert doc.modality_type == "text"
    assert doc.page_number == 1
    assert doc.bounding_box == bbox
    assert doc.source_file == "test_doc.pdf"


def test_semantic_splitter_preserves_atomic_tables_and_images():
    """Verifies that SemanticDocumentSplitter keeps tables and images atomic while recursively splitting text."""
    pipeline = MultimodalIngestionPipeline(chunk_size=400, inject_visual_summaries=True)
    report_pdf = Path("data/uploads/sample_enterprise_report.pdf")
    multi_doc = pipeline.process_file(report_pdf)

    splitter = SemanticDocumentSplitter(chunk_size=300, chunk_overlap=50)
    indexed_docs = splitter.split_document(multi_doc)

    assert len(indexed_docs) >= len(multi_doc.chunks)
    modalities = {d.modality_type for d in indexed_docs}
    assert "text" in modalities
    assert "table" in modalities or "image" in modalities

    # Table chunks must preserve their bounding box and page number
    table_chunks = [d for d in indexed_docs if d.modality_type == "table"]
    for tc in table_chunks:
        assert tc.page_number >= 1
        assert tc.source_file == report_pdf.name


def test_qdrant_vector_store_dense_search(sample_indexed_documents):
    """Verifies local in-memory Qdrant store upsert and dense vector search."""
    qdrant = QdrantVectorStore(in_memory=True)
    count = qdrant.upsert_documents(sample_indexed_documents)
    assert count == len(sample_indexed_documents)
    assert qdrant.count() == len(sample_indexed_documents)

    # Search dense
    results = qdrant.search_dense("operating margin chart", top_k=3)
    assert len(results) > 0
    assert results[0].dense_score is not None


def test_bm25_retriever_exact_keyword_matching(sample_indexed_documents):
    """Verifies BM25 keyword search retrieves exact alphanumeric IDs like TX-1001."""
    bm25 = BM25Retriever()
    bm25.index_documents(sample_indexed_documents)

    results = bm25.search_bm25("TX-1001", top_k=2)
    assert len(results) >= 1
    top = results[0]
    assert "TX-1001" in top.content
    assert top.modality_type == "table"
    assert top.bm25_score > 0.0


def test_hybrid_search_rrf_and_cross_encoder(sample_indexed_documents):
    """
    End-to-End Hybrid Search Test:
    Dense + BM25 ➜ RRF ➜ Top 10 ➜ Cross-Encoder ➜ Top 4 Chunks.
    """
    qdrant = QdrantVectorStore(in_memory=True)
    bm25 = BM25Retriever()
    cross_encoder = CrossEncoderReranker(use_local_fallback=True)

    hybrid = HybridRetriever(
        qdrant_store=qdrant,
        bm25_retriever=bm25,
        cross_encoder=cross_encoder,
        rrf_constant=60,
    )
    hybrid.index_documents(sample_indexed_documents)

    # Query asking for specific alphanumeric transaction in table
    query = "What is the settlement status of TX-1001 in the ledger?"
    final_top4 = hybrid.search(
        query=query,
        top_k_dense=20,
        top_k_bm25=20,
        top_n_rerank=10,
        final_top_k=4,
    )

    # Assert exactly 4 (or up to len(docs)) chunks returned
    assert len(final_top4) <= 4
    assert len(final_top4) > 0

    # The #1 top ranked chunk must be the table containing TX-1001
    top_result = final_top4[0]
    assert top_result.rank == 1
    assert "TX-1001" in top_result.content
    assert top_result.modality_type == "table"
    assert top_result.source_file == "financial_ledger.pdf"
    assert top_result.page_number == 1
    assert top_result.bounding_box is not None
    assert top_result.cross_encoder_score is not None
    assert top_result.rrf_score is not None


def test_bm25_empty_corpus_safety():
    """Verifies BM25Retriever handles empty document indexing and queries safely without ZeroDivisionError."""
    bm25 = BM25Retriever()
    # 1. Indexing empty list must not crash
    bm25.index_documents([])
    assert bm25.count() == 0

    # 2. Searching empty index must return empty list safely
    results = bm25.search_bm25("any query test", top_k=5)
    assert results == []

    # 3. Empty query string on empty index
    assert bm25.search_bm25("", top_k=5) == []


def test_bm25_single_document_positive_scoring():
    """Verifies single-document collections return valid positive scores under Robertson-Lucene formulation."""
    bm25 = BM25Retriever()
    doc = IndexedDocument(
        chunk_id="single_01",
        content="Unique technical specification for aerospace propulsion turbine blades.",
        modality_type="text",
        page_number=1,
        source_file="turbine.pdf",
    )
    bm25.index_documents([doc])
    assert bm25.count() == 1

    # Query matching term in the single document
    results = bm25.search_bm25("turbine propulsion", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "single_01"
    assert results[0].bm25_score > 0.0


def test_bm25_persistence_and_restart_recovery(tmp_path, sample_indexed_documents):
    """Verifies that BM25 index state saves atomically to disk and restores accurately after restart."""
    persist_dir = tmp_path / "bm25_test_index"

    # Instance 1: Index and save
    bm25_inst1 = BM25Retriever(persist_dir=persist_dir, auto_save=True)
    bm25_inst1.index_documents(sample_indexed_documents)
    assert bm25_inst1.count() == len(sample_indexed_documents)

    # Verify files created on disk
    assert (persist_dir / "index_meta.json").exists()
    assert (persist_dir / "corpus_data.json").exists()

    # Simulate process shutdown / restart
    del bm25_inst1

    # Instance 2: Auto-loads from persist_dir
    bm25_inst2 = BM25Retriever(persist_dir=persist_dir)
    assert bm25_inst2.count() == len(sample_indexed_documents)

    # Query loaded index
    results = bm25_inst2.search_bm25("Kubernetes deployment guidelines", top_k=1)
    assert len(results) == 1
    assert "Kubernetes" in results[0].content


def test_synchronized_deletion_and_update(sample_indexed_documents):
    """Verifies delete_chunk, delete_by_source, and update_document keep Qdrant and BM25 synchronized."""
    qdrant = QdrantVectorStore(in_memory=True)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
    hybrid.index_documents(sample_indexed_documents)

    initial_count = len(sample_indexed_documents)
    assert qdrant.count() == initial_count
    assert bm25.count() == initial_count

    # 1. Delete single chunk
    target_cid = "doc1_chunk_0"
    del_ok = hybrid.delete_chunk(target_cid)
    assert del_ok is True
    assert qdrant.count() == initial_count - 1
    assert bm25.count() == initial_count - 1
    assert not qdrant.has_chunk(target_cid)
    assert not bm25.has_chunk(target_cid)

    # 2. Delete by source file
    source_to_delete = "deployment_guide.pdf"
    deleted_source_count = hybrid.delete_by_source(source_to_delete)
    assert deleted_source_count == 1
    assert qdrant.count() == initial_count - 2
    assert bm25.count() == initial_count - 2

    # 3. Public retrieve() method
    retrieve_results = hybrid.retrieve("TX-1001 transaction", top_k=2)
    assert len(retrieve_results) > 0
    assert "TX-1001" in retrieve_results[0].content

