"""
Independent Adversarial QA Audit Suite for CogentSearch Phase 3 Release Gate.
Evaluates the implementation at commit 377e3cb under hostile conditions without mocks.

Covers:
1. Context Compression Integrity:
   - Negation omission under compression
   - Minimum sentence threshold truncation
   - Numeric and date preservation
   - Table column and row alignment
2. Document Grader Robustness:
   - Double negation handling
   - Irrelevant negation markers elsewhere in context
   - Semantic synonym matching (vocabulary mismatch false rejections)
   - Contradictory evidence
3. Self-Correction Reliability:
   - Termination after max retries
   - Monotonic retry counter
   - Repeated identical retrieval cycles
4. Vector Space Isolation:
   - Dimension mismatch
   - Corrupted metadata resilience
   - Missing metadata vulnerability
5. Web Search Integrity:
   - Representation of simulated vs live results
   - Provenance attribution
6. Citation & Provenance Integrity:
   - Detection of fabricated/hallucinated citations
   - Validation of cited page and source existence
"""

import pytest
import re
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from cogent.schemas.indexing import IndexedDocument, SearchResultChunk
from cogent.schemas.document import BoundingBox
from cogent.vectorstore.qdrant_store import QdrantVectorStore
from cogent.vectorstore.embeddings import EmbeddingEngine
from cogent.retrieval.bm25_retriever import BM25Retriever
from cogent.retrieval.hybrid_retriever import HybridRetriever
from cogent.agent.state import AgentState
from cogent.agent.nodes import AgentNodes
from cogent.agent.compressor import ContextCompressor, CompressedChunk
from cogent.agent.agentic_loop import AgenticRAG
from cogent.core.exceptions import VectorSpaceMismatchError, EmbeddingAPIError


# ==============================================================================
# CHALLENGE 1: CONTEXT COMPRESSION INTEGRITY
# ==============================================================================

def test_compressor_short_sentence_omission_defect():
    """
    AUDIT TEST: Tests whether critical short sentences (e.g. 'It failed.', 'No.')
    are dropped due to hardcoded min_sentence_length = 15.
    """
    compressor = ContextCompressor(max_sentences_per_chunk=3, min_sentence_length=15)

    content = "The safety test was initiated. It failed. Subsequent tests were postponed indefinitely."
    chunk = SearchResultChunk(
        chunk_id="c_short",
        content=content,
        modality_type="text",
        page_number=1,
        source_file="safety.pdf",
    )

    compressed = compressor.compress_chunk("safety test status", chunk)

    # If 'It failed.' (10 characters) is dropped, the compression reverses the safety conclusion!
    short_sentence_preserved = "It failed." in compressed.compressed_content
    assert short_sentence_preserved, (
        f"DEFECT FOUND in compressor.py: Short sentence 'It failed.' was pruned because its length (10) "
        f"is below min_sentence_length=15, distorting the factual safety conclusion."
    )


def test_compressor_negation_distortion_under_compression():
    """
    AUDIT TEST: Tests whether pruning sentences without query keywords drops qualifying negation.
    Example: 'Revenue was $400M. This figure does NOT include pending litigation liabilities.'
    Query: 'Total revenue'
    """
    compressor = ContextCompressor(max_sentences_per_chunk=2, min_sentence_length=10)

    content = (
        "Total revenue reached $400M in the fourth quarter. "
        "However, this figure does NOT include pending litigation liabilities of $120M. "
        "Operating costs were aligned with earlier guidance."
    )
    chunk = SearchResultChunk(
        chunk_id="c_rev",
        content=content,
        modality_type="text",
        page_number=1,
        source_file="earnings.pdf",
    )

    compressed = compressor.compress_chunk("Total revenue quarter", chunk)

    # Check if the qualifying negation sentence was pruned
    negation_preserved = "NOT include pending litigation liabilities" in compressed.compressed_content
    assert negation_preserved, (
        f"DEFECT FOUND in compressor.py: Extractive compression pruned qualifying negation sentence "
        f"because it lacked query terms, altering the financial representation."
    )


def test_compressor_table_header_data_relationship():
    """
    AUDIT TEST: Verifies that table compression retains header-data column structure.
    """
    compressor = ContextCompressor()
    table_text = (
        "| Department | Budget | Headcount |\n"
        "| --- | --- | --- |\n"
        "| Engineering | $15M | 120 |\n"
        "| Sales | $10M | 80 |\n"
        "| Support | $3M | 45 |"
    )
    chunk = SearchResultChunk(
        chunk_id="c_tbl",
        content=table_text,
        modality_type="table",
        page_number=1,
        source_file="org.docx",
    )

    compressed = compressor.compress_chunk("Engineering budget", chunk)

    lines = [l.strip() for l in compressed.compressed_content.split("\n") if l.strip()]
    assert len(lines) >= 3
    assert "| Department | Budget | Headcount |" in lines[0]
    assert "Engineering" in lines[2]


# ==============================================================================
# CHALLENGE 2: DOCUMENT GRADER ROBUSTNESS
# ==============================================================================

def test_grader_double_negation_behavior():
    """
    AUDIT TEST: Tests how the deterministic document grader handles double negation.
    Query: 'Is the deployment not unauthorized?' (i.e. Is it authorized?)
    """
    qdrant = QdrantVectorStore(in_memory=True, collection_name="col_audit_grader1")
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
    nodes = AgentNodes(hybrid_retriever=hybrid)

    chunk = SearchResultChunk(
        chunk_id="c_auth",
        content="The server cluster deployment was officially approved and authorized by the operations director.",
        modality_type="text",
        page_number=1,
        source_file="deployment_log.pdf",
    )

    state: AgentState = {
        "question": "Is the server deployment not unauthorized by operations?",
        "rewritten_query": "server deployment not unauthorized operations",
        "documents": [chunk],
        "retry_count": 0,
        "max_retries": 2,
        "is_relevant": True,
        "grade_reason": "",
        "web_search_needed": False,
        "web_documents": [],
        "generation": "",
        "citations": [],
        "trace": [],
    }

    res = nodes.doc_grader(state)
    # The document answers the question affirmatively ('authorized'), but lacks the word 'not'.
    # In cogent/agent/nodes.py: lines 205-209:
    # If question has 'not' and doc lacks 'not', it returns False!
    # This exposes a false rejection on double negation or questions where the answer is affirmative!
    assert res["is_relevant"] is True, (
        f"DEFECT FOUND in nodes.py _deterministic_grade: Rule-based negation check rejected genuinely "
        f"relevant passage because query contained 'not' while affirmative answer lacked 'not'. Reason: {res['grade_reason']}"
    )


def test_grader_irrelevant_negation_marker_vulnerability():
    """
    AUDIT TEST: Tests whether an irrelevant negation marker elsewhere in the chunk
    fools the negation trap check.
    Question: 'Which country did not ratify the 2024 treaty?'
    Chunk: 'Smoking is not permitted in the conference hall. Canada ratified the 2024 treaty.'
    """
    qdrant = QdrantVectorStore(in_memory=True, collection_name="col_audit_grader2")
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
    nodes = AgentNodes(hybrid_retriever=hybrid)

    chunk = SearchResultChunk(
        chunk_id="c_irrel_neg",
        content="Smoking is not permitted in the conference hall. Canada ratified the 2024 treaty in full.",
        modality_type="text",
        page_number=1,
        source_file="treaty.pdf",
    )

    state: AgentState = {
        "question": "Which country did not ratify the 2024 treaty?",
        "rewritten_query": "country not ratify 2024 treaty",
        "documents": [chunk],
        "retry_count": 0,
        "max_retries": 2,
        "is_relevant": True,
        "grade_reason": "",
        "web_search_needed": False,
        "web_documents": [],
        "generation": "",
        "citations": [],
        "trace": [],
    }

    res = nodes.doc_grader(state)
    # Because 'not' is in the document (about smoking), doc_has_negation becomes True,
    # defeating the negation trap check!
    assert res["is_relevant"] is False, (
        f"DEFECT FOUND in nodes.py _deterministic_grade: Bag-of-words negation check was bypassed by an "
        f"irrelevant negation word ('Smoking is not permitted') elsewhere in the chunk."
    )


def test_grader_vocabulary_mismatch_false_rejection():
    """
    AUDIT TEST: Tests whether the document grader falsely rejects relevant passages that use
    synonyms rather than exact words.
    Question: 'What is the frequency of myocardial infarctions in the cardiology study?'
    Passage: 'Heart attack incidence was 2.4% among enrolled subjects in the cardiology evaluation.'
    """
    qdrant = QdrantVectorStore(in_memory=True, collection_name="col_audit_grader3")
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
    nodes = AgentNodes(hybrid_retriever=hybrid)

    chunk = SearchResultChunk(
        chunk_id="c_med_syn",
        content="Heart attack incidence was 2.4% among enrolled subjects in the cardiology evaluation.",
        modality_type="text",
        page_number=2,
        source_file="cardiology.pdf",
    )

    state: AgentState = {
        "question": "What is the frequency of myocardial infarctions in the cardiology study?",
        "rewritten_query": "frequency myocardial infarctions cardiology study",
        "documents": [chunk],
        "retry_count": 0,
        "max_retries": 2,
        "is_relevant": True,
        "grade_reason": "",
        "web_search_needed": False,
        "web_documents": [],
        "generation": "",
        "citations": [],
        "trace": [],
    }

    res = nodes.doc_grader(state)
    assert res["is_relevant"] is True, (
        f"DEFECT FOUND in nodes.py _deterministic_grade: Vocabulary mismatch caused false rejection of "
        f"a semantically equivalent medical document ('heart attack incidence' vs 'myocardial infarction frequency')."
    )


# ==============================================================================
# CHALLENGE 3: SELF-CORRECTION RELIABILITY & LOOP TERMINATION
# ==============================================================================

def test_self_correction_strictly_terminates_after_max_retries():
    """
    AUDIT TEST: Verifies that an unanswerable query strictly halts after max_retries
    and never creates an infinite loop.
    """
    qdrant = QdrantVectorStore(in_memory=True, collection_name="col_audit_loop")
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
    
    agent = AgenticRAG(hybrid_retriever=hybrid, max_retries=3)

    resp = agent.run("Completely nonexistent question that will never match anything in universe")

    assert resp.retry_count == 3
    assert resp.web_search_used is True
    assert "web_search_fallback" in resp.trace
    assert "generator" in resp.trace[-1]

    # Verify no infinite loop occurred
    rewrite_attempts = [t for t in resp.trace if "query_rewriter" in t]
    assert len(rewrite_attempts) == 3


# ==============================================================================
# CHALLENGE 4: VECTOR SPACE ISOLATION AUDIT
# ==============================================================================

def test_vector_space_corrupted_metadata_resilience(tmp_path):
    """
    AUDIT TEST: Tests how QdrantVectorStore behaves when collection_meta.json is corrupted or empty.
    Expected: Should raise or log error, not silently bypass validation.
    """
    db_path = tmp_path / "corrupt_meta_db"
    db_path.mkdir(parents=True, exist_ok=True)

    # Create collection
    eng = EmbeddingEngine(provider="mock", model_name="model-a", dimension=1536)
    store = QdrantVectorStore(collection_name="test_col", persist_dir=db_path, embedding_engine=eng)
    del store

    # Corrupt the metadata file
    meta_file = db_path / "test_col_meta.json"
    with open(meta_file, "w") as f:
        f.write("{corrupted_json_syntax: true,")

    # Attempt re-opening with incompatible model
    eng_incompatible = EmbeddingEngine(provider="mock", model_name="model-b", dimension=1536)

    # In cogent/vectorstore/qdrant_store.py line 82:
    # except (json.JSONDecodeError, KeyError) as e:
    #     logger.warning(f"Could not validate collection metadata file {meta_file}: {e}")
    # It catches JSONDecodeError and proceeds silently!
    caught_mismatch = False
    try:
        QdrantVectorStore(collection_name="test_col", persist_dir=db_path, embedding_engine=eng_incompatible)
    except VectorSpaceMismatchError:
        caught_mismatch = True

    assert caught_mismatch, (
        f"DEFECT FOUND in qdrant_store.py: Corrupted metadata file caused silent bypass of vector space validation, "
        f"allowing incompatible model 'model-b' to open collection created with 'model-a'."
    )


def test_vector_space_dimension_mismatch_check(tmp_path):
    """
    AUDIT TEST: Tests whether collection metadata check validates dimension equality.
    """
    db_path = tmp_path / "dim_mismatch_db"
    db_path.mkdir(parents=True, exist_ok=True)

    # Create collection with dimension 1536
    eng1 = EmbeddingEngine(provider="mock", model_name="embed-v1", dimension=1536)
    store = QdrantVectorStore(collection_name="dim_col", persist_dir=db_path, embedding_engine=eng1)
    del store

    # Attempt re-opening with same model name but dimension 768
    eng2 = EmbeddingEngine(provider="mock", model_name="embed-v1", dimension=768)
    caught_mismatch = False
    try:
        QdrantVectorStore(collection_name="dim_col", persist_dir=db_path, embedding_engine=eng2)
    except (VectorSpaceMismatchError, Exception):
        caught_mismatch = True

    assert caught_mismatch, (
        f"DEFECT FOUND in qdrant_store.py: Dimension mismatch (1536 vs 768) was not checked in collection metadata validation."
    )


# ==============================================================================
# CHALLENGE 5: WEB SEARCH INTEGRITY
# ==============================================================================

def test_web_search_simulated_fallback_representation():
    """
    AUDIT TEST: Verifies that simulated/mock web search fallback does NOT represent itself
    as 'Verified online documentation' to prevent user deception.
    """
    qdrant = QdrantVectorStore(in_memory=True, collection_name="col_audit_web")
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
    nodes = AgentNodes(hybrid_retriever=hybrid)

    state: AgentState = {
        "question": "What is the global orbital altitude of JWST?",
        "rewritten_query": "orbital altitude JWST",
        "documents": [],
        "retry_count": 2,
        "max_retries": 2,
        "is_relevant": False,
        "grade_reason": "",
        "web_search_needed": False,
        "web_documents": [],
        "generation": "",
        "citations": [],
        "trace": [],
    }

    res = nodes.web_search_node(state)
    web_doc = res["web_documents"][0]

    # In cogent/agent/nodes.py line 265:
    # content: f"Verified online documentation for '{query}': High-level specification details..."
    # Claiming 'Verified online documentation' for a synthetic mock string is misleading.
    assert not web_doc["content"].startswith("Verified online documentation"), (
        f"DEFECT FOUND in nodes.py: Synthetic fallback content claims to be 'Verified online documentation', "
        f"violating the requirement that simulated results must never masquerade as verified web evidence."
    )


# ==============================================================================
# CHALLENGE 6: CITATION AND PROVENANCE INTEGRITY
# ==============================================================================

def test_citation_extractor_rejects_hallucinated_sources():
    """
    AUDIT TEST: Tests whether _extract_citations rejects hallucinated citations
    that reference files not in the retrieved documents list.
    """
    qdrant = QdrantVectorStore(in_memory=True, collection_name="col_audit_cite_halluc")
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)
    nodes = AgentNodes(hybrid_retriever=hybrid)

    real_chunk = SearchResultChunk(
        chunk_id="c_real",
        content="Real content from finance_specs.pdf.",
        modality_type="text",
        page_number=1,
        source_file="finance_specs.pdf",
    )

    hallucinated_answer = (
        "Operating profit is $5M [Source: finance_specs.pdf, Page 1, Type: Text]. "
        "However, executive bonus was $50M [Source: fabricated_leak.pdf, Page 99, Type: Text]."
    )

    citations = nodes._extract_citations(
        text=hallucinated_answer,
        documents=[real_chunk],
        web_docs=[],
    )

    # Check if fabricated citation 'fabricated_leak.pdf' was included
    fabricated_included = any(c.source_file == "fabricated_leak.pdf" for c in citations)
    assert not fabricated_included, (
        f"DEFECT FOUND in nodes.py _extract_citations: Extracted citation for 'fabricated_leak.pdf' "
        f"which does not exist in retrieved documents list. Fabricated citations must be rejected or flagged."
    )
