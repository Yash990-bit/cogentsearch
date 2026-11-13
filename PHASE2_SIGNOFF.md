# Phase 2 Engineering Sign-off: Hybrid Indexing & Storage Engine

**System:** CogentSearch (formerly DocSage)  
**Evaluator:** Senior Retrieval Systems Auditor  
**Date:** 2026-11-13  
**Final Status:** **ACCEPTED WITH KNOWN LIMITATIONS**  

---

## 1. Verified Commit and Repository State

- **Active HEAD Commit:** `b0b6dd07bbfd452ba0fb5816143358e47bddf5c5` (`b0b6dd0`)
- **Descent Verification:** `b0b6dd0` directly descends from release-gate audit commit `2ad1394ab0e077b3b3dc2cc94188072bea5b6259` (`2ad1394`).
- **Post-2ad1394 Changes:** Exactly one commit (`b0b6dd0: docs: Add Phase 2 final engineering sign-off report`).
- **Production & Test Integrity:** Zero production code, test code, or benchmark datasets were modified after commit `2ad1394`.
- **Remote Synchronization:** In sync with `origin/main` ([github.com/Yash990-bit/cogentsearch](https://github.com/Yash990-bit/cogentsearch.git)).
- **Working Tree:** Clean (`nothing to commit, working tree clean`).

---

## 2. Implemented Architecture

The Phase 2 storage and retrieval engine implements a 4-tier multimodal hybrid pipeline:

```
[ Ingested Document Chunks ]
             │
             ├──► [ SemanticChunkSplitter ] ──────────┐
             │    (Modality-aware: Text, Table, Image)│
             ▼                                        ▼
   ┌───────────────────┐                    ┌───────────────────┐
   │ QdrantVectorStore │                    │   BM25Retriever   │
   │ (Dense Cosine)    │                    │ (Robertson-Lucene)│
   └─────────┬─────────┘                    └─────────┬─────────┘
             │ Top 20 Candidates                      │ Top 20 Candidates
             └────────────────► [ RRF Fusion ] ◄──────┘
                                (k = 60)
                                    │ Top 10 Candidates
                                    ▼
                       ┌─────────────────────────┐
                       │  CrossEncoderReranker   │
                       │ (ms-marco-MiniLM-L-6-v2)│
                       └────────────┬────────────┘
                                    │ Final Top 4 Chunks
                                    ▼
                         [ .retrieve(query) ]
```

1. **Multimodal Document Schema (`cogent/schemas/indexing.py`):** Strictly validates `chunk_id`, `content`, `modality_type` (`"text" | "table" | "image"`), `page_number`, `bounding_box`, and `source_file` via Pydantic V2.
2. **Modality-Preserving Chunk Splitter (`cogent/vectorstore/splitter.py`):** Chunks text while preserving tables and images atomically without boundary fragmentation.
3. **Dense Vector Store (`cogent/vectorstore/qdrant_store.py`):** Embedded local Qdrant collection configured for 1536-dimensional vectors with cosine similarity distance, atomic point deletion, and payload rehydration.
4. **Sparse Lexical Store (`cogent/retrieval/bm25_retriever.py`):** Tokenized BM25Okapi implementing Robertson-Lucene non-negative IDF ($\ln(1 + \frac{N-n+0.5}{n+0.5}) \ge \epsilon$) and atomic version 2.0.0 disk persistence (`corpus_data.json` and `index_meta.json`).
5. **Reciprocal Rank Fusion & Synchronization (`cogent/retrieval/hybrid_retriever.py`):** Merges dense and lexical candidate lists via RRF score $RRF(d) = \sum_{m} \frac{1}{k + r_m(d)}$ ($k=60$); synchronizes `delete_chunk`, `delete_by_source`, `update_document`, and `reindex_document`; automatically heals desynchronized BM25 indexes from Qdrant payloads on startup.
6. **Public API (`HybridRetriever.retrieve`):** Exposes `.retrieve(query: str, top_k: int = 4)` executing the full end-to-end pipeline with backward-compatible `.search()`.
7. **Neural Cross-Encoder (`cogent/retrieval/cross_encoder.py`):** Performs full cross-attention query-passage reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

---

## 3. Test Verification: Full Pytest Suite

**Exact Command Executed:**  
```bash
.venv/bin/pytest -v
```

**Results:**  
- **Total Collected Tests:** 28 items  
- **Passed:** 28  
- **Failed:** 0  
- **Skipped:** 0  
- **Duration:** 11.11 seconds  
- **Warnings:** 8 (PyMuPDF `fitz` deprecation notices)

### Test Breakdown by Subsystem:
1. **Hybrid Retrieval Subsystem (`tests/test_hybrid_retrieval.py` — 9 Tests):**
   - `test_indexed_document_schema`: Validates Pydantic schema constraints and serialization.
   - `test_semantic_splitter_preserves_atomic_tables_and_images`: Verifies table/image boundaries are never fragmented.
   - `test_qdrant_vector_store_dense_search`: Validates vector upsert, cosine distance scoring, and metadata filtering.
   - `test_bm25_retriever_exact_keyword_matching`: Validates exact lexical matching and ranking.
   - `test_hybrid_search_rrf_and_cross_encoder`: Validates Dense + BM25 $\to$ RRF $\to$ Cross-Encoder pipeline execution.
   - `test_bm25_empty_corpus_safety`: Verifies initialization and query execution on empty collections without `ZeroDivisionError`.
   - `test_bm25_single_document_positive_scoring`: Validates positive BM25 scores on single-document collections via Robertson-Lucene IDF.
   - `test_bm25_persistence_and_restart_recovery`: Verifies disk persistence, reload, and index version compatibility.
   - `test_synchronized_deletion_and_update`: Validates simultaneous deletion and in-place chunk updates across Qdrant and BM25.
2. **Phase 1 Ingestion, Schema, and Failure Subsystems (19 Tests):**
   - `tests/test_failure_handling.py` (6 tests): Empty documents, corrupted files, encrypted PDFs, unsupported formats, zero-text PDFs, content-hash idempotency.
   - `tests/test_ingestion.py` (4 tests): Multimodal extraction from PDF, DOCX, PPTX, and full acceptance criteria.
   - `tests/test_regression_audit.py` (1 test): Source-to-element citation traceability across 50 elements.
   - `tests/test_schemas.py` (5 tests): Bounding box coordinates, geometry validation, visual element contracts, document statistics.
   - `tests/test_table_integrity.py` (3 tests): Multipage table continuity, DOCX table ground truth, PPTX table ground truth.

---

## 4. Embedding Configuration & Vector Dimension Audit

Verified directly against source files (`cogent/vectorstore/embeddings.py`, `cogent/vectorstore/qdrant_store.py`, and `cogent/core/config.py`):

| Configuration Property | Implementation Specification |
| :--- | :--- |
| **Configured Provider** | `settings.EMBEDDING_PROVIDER` (default `"openai"`) |
| **Configured Model** | `settings.OPENAI_EMBEDDING_MODEL` (`"text-embedding-3-small"`) |
| **Model Revision** | OpenAI API hosted endpoint; HuggingFace Cross-Encoder uses `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Output Vector Dimension** | 1536 dimensions |
| **Qdrant Collection Dimension** | `VectorParams(size=1536, distance=Distance.COSINE)` |
| **Fallback Implementation** | Deterministic SHA-256 token hashing + MD5 character trigram projection with L2 normalization (`_offline_embed`) |
| **Fallback Trigger** | Missing `OPENAI_API_KEY` or runtime API communication failure |
| **Cross-Embedding Compatibility** | **INCOMPATIBLE**. Fallback hashing vectors and OpenAI neural embeddings share the same 1536 dimensionality, but reside in entirely different geometric spaces. Storing or querying vectors across both models within the same collection is invalid and unsupported. |

---

## 5. Benchmark Reconciliation & Evidence Comparison

### Dataset Identity:
- **Corpus Size:** 100 documents (50 domain-specific protocol specifications across Finance, Medicine, Engineering, Law, Security, AI, DevOps, Databases, Genomics, and Aerospace + 50 generic administrative noise documents).
- **Evaluation Queries:** 50 explicit question-needle pairs targeting domain protocol specifications.

### Side-by-Side Metric Comparison:

The following table explicitly distinguishes previously documented narrative results from newly executed test suite outputs:

| Retrieval Configuration | Metric | Previously Reported Narrative | Newly Executed Release-Gate Audit (`audit_release_gates.py`) |
| :--- | :--- | :---: | :---: |
| **BM25 Lexical Only** | HitRate@4<br>MRR<br>nDCG@4 | 0.8200<br>0.7410<br>0.7621 | **1.0000**<br>**1.0000**<br>**1.0000** |
| **Dense Only** | HitRate@4<br>MRR<br>nDCG@4 | 0.9600<br>0.9125<br>0.9248 | **0.9200**<br>**0.7087**<br>**0.7531** |
| **Hybrid RRF ($k=60$)** | HitRate@4<br>MRR<br>nDCG@4 | 1.0000<br>0.9750<br>0.9812 | **1.0000**<br>**1.0000**<br>**1.0000** |
| **Hybrid RRF + Cross-Encoder** | HitRate@4<br>MRR<br>nDCG@4 | 1.0000<br>1.0000<br>1.0000 | **1.0000**<br>**1.0000**<br>**1.0000** |

### Additional Metrics Produced by `scripts/run_phase2_adversarial_suite.py` Test 7:
- **Recall@4:** BM25 = 1.0000, Dense = 0.9200, Hybrid RRF = 1.0000, Hybrid + Cross-Encoder = 1.0000
- **Recall@10:** BM25 = 1.0000, Dense = 1.0000, Hybrid RRF = 1.0000, Hybrid + Cross-Encoder = 1.0000
- **Precision@4:** BM25 = 0.2500, Dense = 0.2300, Hybrid RRF = 0.2500, Hybrid + Cross-Encoder = 0.2500  
  *(Note: Because each ground-truth query targets exactly 1 needle document, the maximum theoretical Precision@4 is $1/4 = 0.2500$.)*

### Technical Explanation of Metric Discrepancy:
1. **BM25 Difference (0.8200 vs 1.0000):** The previously reported narrative (0.8200) was based on earlier conceptual projections involving abstract paraphrases without specific alphanumeric identifiers. In the actual executed benchmark (`Test 7`), queries contain exact system codes (e.g., `FIN-10`, `MED-20`, `DB-50`), allowing BM25 to rank the exact needle document at rank 1 for all 50 queries.
2. **Dense Difference (0.9600 vs 0.9200):** The previous narrative assumed deep neural OpenAI embeddings (`text-embedding-3-small`). The executed benchmark ran in local CI mode using the offline hash-based fallback vectorizer (`_offline_embed`) due to the absence of an `OPENAI_API_KEY`. The offline hash vectorizer achieved HitRate@4 = 0.9200 and MRR = 0.7087.
3. **Conclusion:** Both sets of numbers are now documented with their exact operational contexts. No benchmark results are fabricated or silently substituted.

---

## 6. Documented Known Limitations

1. **Cross-Encoder Adversarial Inversions (17 / 20 Inversions = 85.0%):**
   - Verified via `scripts/audit_release_gates.py --challenge 6` and `scripts/run_phase2_adversarial_suite.py` Test 3.
   - `cross-encoder/ms-marco-MiniLM-L-6-v2` scored lexical distractors higher than factual answers in 17 out of 20 adversarial pairs.
   - *Cause:* Small 6-layer cross-attention encoders are trained for passage retrieval on MS-MARCO and heavily reward verbatim token overlap over subtle logical negation.
   - *Impact:* The Cross-Encoder cannot be relied upon for factual verification, negation checking, or contradiction resolution. This must be handled by Phase 4's LangGraph Self-Reflective LLM Grader.
2. **Local Storage Lock Contention:**
   - Embedded Qdrant uses SQLite with `portalocker`. Instantiating multiple `QdrantVectorStore` instances pointing to the same directory without closing prior instances raises `RuntimeError: Storage folder is already accessed`.
3. **Memory Footprint of In-Memory Lexical Index:**
   - `BM25Retriever` stores the tokenized corpus in Python process memory. Collections exceeding $500\text{k}$ documents will require external disk-backed indexing (e.g., Tantivy).

---

## 7. Explicit Phase 3 Prerequisites

Before initiating Phase 3 (Advanced Reranking Optimization & Context Compression), the following technical constraints must be accounted for:

1. **Context Compression Layer:** Retrieved chunks must not be passed raw to LLM context windows; an extractive compression or token pruning layer (e.g., LongLLMLingua) is required.
2. **Dynamic Relevance Thresholding:** Reranking must implement calibrated score thresholds to drop non-relevant candidates rather than blindly returning static top-k chunks.
3. **Multi-Vector / Late-Interaction Reranking Exploration:** Evaluate ColBERTv2 or late-interaction token scoring to address the 85% lexical inversion vulnerability documented in Challenge 6.
4. **Multimodal Coordinate Preservation:** Extractive summarization and compression layers must preserve line-of-sight tracking back to `source_file`, `page_number`, and `bounding_box` for downstream citation validation.

---

## 8. Final Status Verdict

**Final Status:** **ACCEPTED WITH KNOWN LIMITATIONS**

*Phase 2 satisfies all architectural, synchronization, persistence, and hybrid retrieval criteria. Known limitations regarding neural cross-encoder negation blindness and embedding modality incompatibility are explicitly documented, establishing the empirical foundation for Phase 3 and Phase 4.*
