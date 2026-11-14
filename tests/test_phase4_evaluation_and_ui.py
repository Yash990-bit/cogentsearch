"""
Unit & Integration Tests for CogentSearch Phase 4: Evaluation & UI Setup.
Verifies:
1. Pydantic structured output schemas for Generator.
2. Smart cost-efficient model routing configuration (gpt-4o-mini for Rewriter/Grader, gpt-4o for Generator).
3. RAGAS evaluation metrics computation (Faithfulness and Context Precision).
4. Streamlit UI bounding box image crop generation and multimodal reference display.
"""

import pytest
from pathlib import Path
from cogent.core.config import settings
from cogent.schemas.indexing import SearchResultChunk
from cogent.schemas.document import BoundingBox
from cogent.agent.state import StructuredCitation, GeneratorStructuredOutput, CitationItem
from cogent.agent.nodes import AgentNodes
from cogent.vectorstore.qdrant_store import QdrantVectorStore
from cogent.retrieval.bm25_retriever import BM25Retriever
from cogent.retrieval.hybrid_retriever import HybridRetriever
from scripts.evaluate_ragas import compute_deterministic_ragas_metrics, BENCHMARK_QA_PAIRS
from app import get_bounding_box_crop


def test_smart_model_routing_configuration():
    """Verifies that cost-effective smart model routing is configured."""
    assert settings.REWRITER_MODEL_NAME == "gpt-4o-mini", "Query Rewriter should use cheap gpt-4o-mini"
    assert settings.GRADER_MODEL_NAME == "gpt-4o-mini", "Document Grader should use cheap gpt-4o-mini"
    assert settings.GENERATOR_MODEL_NAME == "gpt-4o", "Answer Generator should use higher-capacity model"


def test_generator_pydantic_structured_output_schema():
    """Verifies Pydantic structured output models for OpenAI function calling."""
    citation = StructuredCitation(
        source_file="earnings_report.docx",
        page_number=3,
        modality_type="Table",
        cited_claim="Q1 revenue was $10M",
        bounding_box={"x0": 60, "y0": 150, "x1": 550, "y1": 350},
    )
    assert citation.source_file == "earnings_report.docx"
    assert citation.page_number == 3
    assert citation.modality_type == "Table"

    structured_output = GeneratorStructuredOutput(
        answer="According to verified earnings reports, Q1 revenue was $10M [Source: earnings_report.docx, Page 3, Type: Table].",
        citations=[citation],
    )
    assert len(structured_output.citations) == 1
    assert "earnings_report.docx" in structured_output.answer


def test_structured_citation_grounding_validation():
    """Verifies that _validate_structured_citations rejects hallucinated sources."""
    qdrant = QdrantVectorStore(in_memory=True, collection_name="col_test_struct_cite")
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
    nodes = AgentNodes(hybrid_retriever=hybrid)

    real_chunk = SearchResultChunk(
        chunk_id="c_real",
        content="Real protocol specification.",
        modality_type="text",
        page_number=1,
        source_file="spec_v1.pdf",
        bounding_box=BoundingBox(page_number=1, x0=50, y0=50, x1=400, y1=200, page_width=600, page_height=800),
    )

    raw_citations = [
        StructuredCitation(source_file="spec_v1.pdf", page_number=1, modality_type="Text"),
        StructuredCitation(source_file="fake_leak.pdf", page_number=99, modality_type="Text"),
    ]

    validated = nodes._validate_structured_citations(raw_citations, documents=[real_chunk], web_docs=[])

    assert len(validated) == 1
    assert validated[0].source_file == "spec_v1.pdf"
    assert validated[0].bounding_box is not None
    assert not any(c.source_file == "fake_leak.pdf" for c in validated)


def test_ragas_metrics_computation():
    """Verifies Faithfulness and Context Precision computation."""
    question = "What is the capital expenditure threshold in FIN-10?"
    ground_truth = "Capital expenditure threshold in protocol FIN-10 is $25M."
    contexts = [
        "Protocol FIN-10 specifies operating margin parameters and quarterly capital expenditure thresholds of $25M.",
        "Other unrelated administrative guidelines.",
    ]
    supported_answer = "Protocol FIN-10 specifies capital expenditure thresholds of $25M [Source: fin.pdf, Page 1, Type: Text]."

    faithfulness, context_precision = compute_deterministic_ragas_metrics(
        question=question,
        answer=supported_answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )

    assert faithfulness == 1.0, f"Expected 1.0 faithfulness, got {faithfulness}"
    assert context_precision == 1.0, f"Expected 1.0 context precision, got {context_precision}"


def test_ragas_benchmark_qa_dataset_size():
    """Verifies that 10 benchmark QA pairs are defined covering multimodal formats."""
    assert len(BENCHMARK_QA_PAIRS) == 10
    categories = [item["category"] for item in BENCHMARK_QA_PAIRS]
    assert any("Table" in cat for cat in categories)
    assert any("Diagram" in cat or "Architecture" in cat for cat in categories)
    assert any("Medical" in cat or "Clinical" in cat for cat in categories)


def test_get_bounding_box_crop_pdf():
    """Verifies that get_bounding_box_crop produces valid PNG image bytes from a real test PDF."""
    pdf_path = Path("data/test_corpus/doc_02_multipage_table.pdf")
    if pdf_path.exists():
        crop_bytes = get_bounding_box_crop(
            file_path=pdf_path,
            page_number=1,
            bbox={"x0": 60, "y0": 117, "x1": 417, "y1": 207},
        )
        assert crop_bytes is not None
        assert crop_bytes.startswith(b"\x89PNG")  # PNG signature


def test_chainlit_document_ingestion_helper():
    """Verifies that ingest_document_file processes and indexes documents for Chainlit session."""
    from app import ingest_document_file
    from cogent.ingestion.pipeline import MultimodalIngestionPipeline
    from cogent.vectorstore.splitter import SemanticDocumentSplitter
    
    qdrant = QdrantVectorStore(in_memory=True, collection_name="col_test_chainlit_ingest")
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
    
    pdf_path = Path("data/test_corpus/doc_02_multipage_table.pdf")
    if pdf_path.exists():
        doc, chunks = ingest_document_file(
            file_path=pdf_path,
            pipeline=MultimodalIngestionPipeline(),
            splitter=SemanticDocumentSplitter(),
            retriever=hybrid,
        )
        assert doc is not None
        assert len(chunks) > 0
        assert hybrid.qdrant_store.count() > 0
        assert hybrid.bm25_retriever.count() > 0


def test_chainlit_step_vector_search_robustness():
    """Verifies that format_vector_search_step handles SearchResultChunk attributes without AttributeError."""
    from app import format_vector_search_step
    chunk = SearchResultChunk(
        chunk_id="c_test",
        content="Test content",
        modality_type="table",
        page_number=2,
        source_file="doc_sample.pdf",
        rrf_score=0.016,
        cross_encoder_score=0.985,
    )
    result = format_vector_search_step(query="financial metrics", documents=[chunk])
    assert "TABLE" in result
    assert "0.0160" in result
    assert "0.9850" in result



