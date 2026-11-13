"""
Unit and Integration Tests for CogentSearch Phase 3: LangGraph Agentic Loop.
Verifies:
1. Query Rewriter optimization.
2. Hybrid Search execution inside retriever node.
3. Document Grader self-correction and relevance verification.
4. Grader conditional routing (to Generator, loop back to Rewriter, or fallback to Web Search).
5. Strict citation enforcement: [Source: FileName, Page X, Type: Table/Text].
6. Complete end-to-end LangGraph execution.
"""

import pytest
import re
from typing import List

from cogent.schemas.indexing import IndexedDocument, SearchResultChunk
from cogent.schemas.document import BoundingBox
from cogent.vectorstore.qdrant_store import QdrantVectorStore
from cogent.retrieval.bm25_retriever import BM25Retriever
from cogent.retrieval.hybrid_retriever import HybridRetriever
from cogent.agent.state import AgentState
from cogent.agent.nodes import AgentNodes
from cogent.agent.graph import build_agentic_rag_graph, route_after_grading
from cogent.agent.agentic_loop import AgenticRAG


@pytest.fixture
def test_retriever():
    """Sets up an in-memory HybridRetriever with sample multimodal chunks."""
    qdrant = QdrantVectorStore(in_memory=True, collection_name="test_agent_col")
    bm25 = BM25Retriever()

    docs = [
        IndexedDocument(
            chunk_id="c_fin_01",
            content="Protocol FIN-10 specifies operating margin parameters and quarterly capital expenditure thresholds.",
            modality_type="text",
            page_number=1,
            bounding_box=BoundingBox(page_number=1, x0=50, y0=100, x1=500, y1=200, page_width=600, page_height=800),
            source_file="finance_specs.pdf",
        ),
        IndexedDocument(
            chunk_id="c_fin_02",
            content="| Metric | Q1 | Q2 | Q3 | Q4 |\n| Revenue | $10M | $12M | $14M | $18M |\n| EBITDA | $2.5M | $3.1M | $3.8M | $4.5M |",
            modality_type="table",
            page_number=3,
            bounding_box=BoundingBox(page_number=3, x0=60, y0=150, x1=550, y1=350, page_width=600, page_height=800),
            source_file="earnings_report.docx",
        ),
        IndexedDocument(
            chunk_id="c_med_01",
            content="Clinical trial dosage protocol MED-20 double-blind placebo control efficacy study.",
            modality_type="text",
            page_number=5,
            bounding_box=BoundingBox(page_number=5, x0=40, y0=80, x1=520, y1=220, page_width=600, page_height=800),
            source_file="clinical_trial.pdf",
        ),
    ]

    hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
    hybrid.index_documents(docs)
    return hybrid


def test_query_rewriter_node(test_retriever):
    """Verifies that conversational queries are stripped and optimized."""
    nodes = AgentNodes(hybrid_retriever=test_retriever)

    initial_state: AgentState = {
        "question": "Can you please find the specific parameters for FIN-10 protocol?",
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

    result = nodes.query_rewriter(initial_state)

    assert "rewritten_query" in result
    assert result["retry_count"] == 1
    # Check that conversational prefixes are removed
    assert not result["rewritten_query"].lower().startswith("can you please find")
    assert "fin-10" in result["rewritten_query"].lower()
    assert "query_rewriter (attempt 1)" in result["trace"]


def test_retriever_node_executes_hybrid_search(test_retriever):
    """Verifies that retriever_node executes Phase 2 hybrid search."""
    nodes = AgentNodes(hybrid_retriever=test_retriever)

    state: AgentState = {
        "question": "FIN-10 protocol specifications",
        "rewritten_query": "FIN-10 protocol specifications",
        "documents": [],
        "retry_count": 1,
        "max_retries": 2,
        "is_relevant": False,
        "grade_reason": "",
        "web_search_needed": False,
        "web_documents": [],
        "generation": "",
        "citations": [],
        "trace": [],
    }

    result = nodes.retriever_node(state)

    assert len(result["documents"]) > 0
    top_doc = result["documents"][0]
    assert isinstance(top_doc, SearchResultChunk)
    assert top_doc.source_file == "finance_specs.pdf"
    assert "retriever" in result["trace"]


def test_doc_grader_relevance_assessment(test_retriever):
    """Verifies that doc_grader correctly scores relevant and irrelevant chunks."""
    nodes = AgentNodes(hybrid_retriever=test_retriever)

    # 1. Relevant context test
    rel_chunk = SearchResultChunk(
        chunk_id="c_fin_01",
        content="Protocol FIN-10 specifies operating margin parameters and quarterly capital expenditure thresholds.",
        modality_type="text",
        page_number=1,
        source_file="finance_specs.pdf",
        score=0.95,
    )

    state_rel: AgentState = {
        "question": "What are the operating margin parameters for FIN-10?",
        "rewritten_query": "operating margin parameters FIN-10",
        "documents": [rel_chunk],
        "retry_count": 1,
        "max_retries": 2,
        "is_relevant": False,
        "grade_reason": "",
        "web_search_needed": False,
        "web_documents": [],
        "generation": "",
        "citations": [],
        "trace": [],
    }

    res_rel = nodes.doc_grader(state_rel)
    assert res_rel["is_relevant"] is True
    assert "doc_grader" in res_rel["trace"]

    # 2. Irrelevant context test
    irrel_chunk = SearchResultChunk(
        chunk_id="c_other",
        content="The cafeteria serves soup and sandwiches every Wednesday at noon.",
        modality_type="text",
        page_number=10,
        source_file="office.pdf",
        score=0.20,
    )

    state_irrel: AgentState = {
        "question": "What are the orbital mechanics for Mars insertion?",
        "rewritten_query": "orbital mechanics Mars insertion",
        "documents": [irrel_chunk],
        "retry_count": 1,
        "max_retries": 2,
        "is_relevant": False,
        "grade_reason": "",
        "web_search_needed": False,
        "web_documents": [],
        "generation": "",
        "citations": [],
        "trace": [],
    }

    res_irrel = nodes.doc_grader(state_irrel)
    assert res_irrel["is_relevant"] is False


def test_conditional_routing_logic():
    """Verifies route_after_grading edge transitions."""
    # Case 1: Relevant -> generator
    state_pass: AgentState = {
        "is_relevant": True,
        "retry_count": 1,
        "max_retries": 2,
    }  # type: ignore
    assert route_after_grading(state_pass) == "generator"

    # Case 2: Irrelevant and retries remain -> loop back to query_rewriter
    state_retry: AgentState = {
        "is_relevant": False,
        "retry_count": 1,
        "max_retries": 2,
    }  # type: ignore
    assert route_after_grading(state_retry) == "query_rewriter"

    # Case 3: Irrelevant and retries exhausted -> fallback to web_search
    state_fallback: AgentState = {
        "is_relevant": False,
        "retry_count": 2,
        "max_retries": 2,
    }  # type: ignore
    assert route_after_grading(state_fallback) == "web_search"


def test_generator_strict_citation_enforcement(test_retriever):
    """Verifies that Generator strictly adheres to [Source: FileName, Page X, Type: Table/Text]."""
    nodes = AgentNodes(hybrid_retriever=test_retriever)

    table_chunk = SearchResultChunk(
        chunk_id="c_fin_02",
        content="| Metric | Q1 | Q2 |\n| Revenue | $10M | $12M |",
        modality_type="table",
        page_number=3,
        source_file="earnings_report.docx",
        score=0.98,
        bounding_box=BoundingBox(page_number=3, x0=60, y0=150, x1=550, y1=350, page_width=600, page_height=800),
    )

    state: AgentState = {
        "question": "What was the Q1 revenue?",
        "rewritten_query": "Q1 revenue earnings report",
        "documents": [table_chunk],
        "retry_count": 1,
        "max_retries": 2,
        "is_relevant": True,
        "grade_reason": "Contains exact revenue numbers.",
        "web_search_needed": False,
        "web_documents": [],
        "generation": "",
        "citations": [],
        "trace": [],
    }

    result = nodes.generator_node(state)

    generation = result["generation"]
    citations = result["citations"]

    # Verify citation tag format
    citation_pattern = r"\[Source:\s*([^,]+),\s*Page\s*(\d+),\s*Type:\s*([^\]]+)\]"
    matches = re.findall(citation_pattern, generation)

    assert len(matches) > 0
    assert matches[0][0] == "earnings_report.docx"
    assert matches[0][1] == "3"
    assert matches[0][2].lower() == "table"

    # Verify structured citations
    assert len(citations) > 0
    assert citations[0]["source_file"] == "earnings_report.docx"
    assert citations[0]["page_number"] == 3
    assert citations[0]["modality_type"].lower() == "table"
    assert citations[0]["bounding_box"] is not None


def test_end_to_end_agentic_rag_direct_hit(test_retriever):
    """Verifies the complete LangGraph StateGraph execution on a direct hit query."""
    agent = AgenticRAG(hybrid_retriever=test_retriever, max_retries=2)

    response = agent.run("What parameters govern FIN-10 protocol?")

    assert response.question == "What parameters govern FIN-10 protocol?"
    assert len(response.rewritten_query) > 0
    assert response.retrieved_chunk_count > 0
    assert response.web_search_used is False
    assert len(response.citations) > 0
    assert any("finance_specs.pdf" in c.source_file for c in response.citations)

    # Trace audit: must have traversed rewriter -> retriever -> doc_grader -> generator
    trace_str = " -> ".join(response.trace)
    assert "query_rewriter (attempt 1)" in trace_str
    assert "retriever" in trace_str
    assert "doc_grader" in trace_str
    assert "generator" in trace_str


def test_end_to_end_agentic_rag_with_web_search_fallback(test_retriever):
    """Verifies that an unknown query loops through rewrite and triggers web search fallback."""
    agent = AgenticRAG(hybrid_retriever=test_retriever, max_retries=2)

    # Query about something completely absent from local index
    response = agent.run("What is the quantum telemetry frequency for Voyager 1 spacecraft?")

    assert response.web_search_used is True
    assert response.retry_count == 2
    assert len(response.citations) > 0
    assert any("web" in c.modality_type.lower() or "web" in c.source_file.lower() for c in response.citations)

    trace_str = " -> ".join(response.trace)
    assert "web_search_fallback" in trace_str
    assert "generator" in trace_str


def test_context_compressor_text_table_image():
    """Verifies that ContextCompressor extracts relevant sentences and table rows while preserving provenance."""
    from cogent.agent.compressor import ContextCompressor

    compressor = ContextCompressor(max_sentences_per_chunk=2)

    # 1. Text chunk test
    chunk_text = SearchResultChunk(
        chunk_id="c_txt",
        content="General background introduction. The capital expenditure budget is set to $45M for Q3. Weather forecast remains sunny. Operational costs are stable.",
        modality_type="text",
        page_number=2,
        source_file="budget.pdf",
        score=0.9,
    )
    res_text = compressor.compress_chunk("capital expenditure budget Q3", chunk_text)
    assert "capital expenditure budget" in res_text.compressed_content
    assert res_text.source_file == "budget.pdf"
    assert res_text.page_number == 2
    assert res_text.compression_ratio < 1.0

    # 2. Table chunk test
    table_content = "| Department | Allocation | Status |\n| --- | --- | --- |\n| Engineering | $20M | Approved |\n| Marketing | $5M | Pending |\n| Legal | $2M | Approved |"
    chunk_table = SearchResultChunk(
        chunk_id="c_tbl",
        content=table_content,
        modality_type="table",
        page_number=4,
        source_file="budget_table.docx",
        score=0.95,
    )
    res_table = compressor.compress_chunk("Engineering allocation", chunk_table)
    assert "Engineering" in res_table.compressed_content
    assert "| Department | Allocation | Status |" in res_table.compressed_content


def test_doc_grader_negation_lexical_trap_rejection(test_retriever):
    """Verifies that Document Grader detects and rejects negation lexical traps that fool cross-encoders."""
    nodes = AgentNodes(hybrid_retriever=test_retriever)

    # Question asking for negation
    q = "Which country did not ratify the 2024 environmental treaty?"

    # Distractor passage with high verbatim overlap but affirmative statement
    distractor_chunk = SearchResultChunk(
        chunk_id="c_trap",
        content="Canada ratified the 2024 environmental treaty in full compliance with international environmental standards.",
        modality_type="text",
        page_number=1,
        source_file="treaty_status.pdf",
        score=0.99,
    )

    state: AgentState = {
        "question": q,
        "rewritten_query": "country not ratify 2024 environmental treaty",
        "documents": [distractor_chunk],
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

    result = nodes.doc_grader(state)
    assert result["is_relevant"] is False
    assert "lexical trap detected" in result["grade_reason"]


def test_vector_space_mismatch_guard(tmp_path):
    """Verifies that VectorSpaceMismatchError is raised if an existing collection is accessed with mismatched embedding config."""
    from cogent.core.exceptions import VectorSpaceMismatchError
    from cogent.vectorstore.embeddings import EmbeddingEngine

    db_dir = tmp_path / "qdrant_isolation_test"
    db_dir.mkdir(parents=True, exist_ok=True)

    # Initialize collection with mock provider
    engine1 = EmbeddingEngine(provider="mock", model_name="mock-model", dimension=1536)
    store1 = QdrantVectorStore(collection_name="col_iso", persist_dir=db_dir, embedding_engine=engine1)
    del store1

    # Attempt to re-open the same collection with a different provider/model
    engine2 = EmbeddingEngine(provider="mock", model_name="different-transformer-v2", dimension=1536)
    with pytest.raises(VectorSpaceMismatchError) as exc_info:
        QdrantVectorStore(collection_name="col_iso", persist_dir=db_dir, embedding_engine=engine2)

    assert "different embedding spaces" in str(exc_info.value)


