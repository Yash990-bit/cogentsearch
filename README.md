# CogentSearch: Multimodal Agentic Document Intelligence Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/architecture-Agentic_RAG-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/tests-19%2F19%20passed%20(100%25)-brightgreen.svg)](#testing--verification)
[![Citation Audit](https://img.shields.io/badge/citation_audit-50%2F50%20verified-success.svg)](#citation-integrity-audit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CogentSearch** is an enterprise-grade, multimodal agentic document intelligence system built from scratch. It solves the critical bottleneck of enterprise search: **retrieving and grounding information trapped in complex tables, architectural diagrams, multi-column layouts, charts, and slide decks** without hallucinations.

Designed with rigorous backend engineering, strict Pydantic V2 data contracts, and complete source-to-chunk provenance linking.

---

## 🏛️ System Architecture

```text
[ Document: PDF / DOCX / PPTX ]
               │
               ▼
   [ Universal Layout Parser ]
   (PyMuPDF / Docling Layout Engine)
   - Estimates body font sizes to detect heading trees (#, ##, ###)
   - Resolves multi-column reading order (left-to-right, top-to-bottom)
   - Detects rotated pages & multi-page continuous tables
               │
       ┌───────┴──────────────────────────────────────────┐
       ▼                                                  ▼
[ Text Flow & Headers ]                        [ Visual & Tabular Elements ]
(Page numbers, Hierarchy)                      (Bounding Boxes: [x0, y0, x1, y1])
       │                                                  │
       │                                                  ▼
       │                                      [ High-DPI Matrix Cropper ]
       │                                      (2.5x matrix rendering)
       │                                                  │
       │                                                  ▼
       │                                      [ Vision LLM Reasoning Engine ]
       │                                      (GPT-4o-mini / Claude 3.5 Sonnet)
       │                                      - Deep semantic summaries
       │                                      - Reconstructed Markdown tables
       │                                      - Search keyword tags
       │                                                  │
       └─────────────────────────┬────────────────────────┘
                                 ▼
              [ Enriched Multimodal Document Schema ]
              - Strict BoundingBox coordinate bounds
              - Verbatim extracted text preserved separately from LLM descriptions
              - Full provenance linking from chunks to visual assets
```

---

## 🛠️ Core Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Ingestion Engine** | `PyMuPDF (fitz)`, `python-docx`, `python-pptx`, `pdfplumber` | Format parsers, coordinate localization, high-DPI cropping |
| **Vision Intelligence** | OpenAI `gpt-4o-mini` / Anthropic `claude-3-5-sonnet` | Visual semantics, chart trend analysis, Markdown table reconstruction |
| **Data Contracts** | `Pydantic V2`, `Pydantic Settings` | Strict geometric coordinate bounds, immutable types, full serialization |
| **Vector Indexing** *(Phase 2)* | Dense (`text-embedding-3-small`) + Sparse (`BGE-M3`) | Hybrid lexical-semantic search |
| **Vector Database** *(Phase 2)* | `Qdrant` / `ChromaDB` | Persistent payload vector storage with metadata filtering |
| **Reranking** *(Phase 3)* | `cross-encoder/ms-marco-MiniLM-L-6-v2` / `Cohere Rerank` | Precision reranking and context compression |
| **Agentic Loop** *(Phase 4)* | `LangGraph` | State machine with hallucination grader, query router, self-correction |
| **Evaluation** *(Phase 5)* | `RAGAS` | Automated benchmarking (Faithfulness, Precision, Answer Relevance) |
| **User Interface** *(Phase 6)* | `Streamlit` | Interactive document workspace with side-by-side bounding box highlights |

---

## 📁 Repository Structure

```text
CogentSearch/
├── .env.example                # Centralized environment & API configuration
├── requirements.txt            # Explicit dependency declarations
├── run_ingestion.py            # Rich CLI interface for document ingestion
├── CITATION_AUDIT_REPORT.md    # Auditable report of 50 randomly sampled elements
├── cogent/
│   ├── core/
│   │   ├── config.py           # Pydantic Settings & loguru configuration
│   │   └── exceptions.py       # Domain-specific typed exceptions
│   ├── schemas/
│   │   └── document.py         # Pydantic V2 data contracts (BoundingBox, Chunks, etc.)
│   ├── ingestion/
│   │   ├── parser.py           # Universal parsers (PDF, DOCX, PPTX)
│   │   └── pipeline.py         # End-to-end ingestion orchestrator with retries
│   ├── vision/
│   │   ├── llm_client.py       # Multi-provider Vision LLM client (OpenAI, Anthropic, Mock)
│   │   └── extractor.py        # Table/image detector & high-DPI cropper
│   ├── vectorstore/            # Phase 2: Hybrid vector store connectors
│   ├── agent/                  # Phase 4: LangGraph state machine & tools
│   ├── evaluation/             # Phase 5: RAGAS automated evaluator
│   └── ui/                     # Phase 6: Streamlit interactive interface
├── data/
│   ├── test_corpus/            # 27-document ground-truth test corpus
│   │   ├── manifest.json       # Ground-truth metadata & expected values
│   │   └── benchmark_results.json # Multi-threaded performance metrics
│   ├── extracted_images/       # High-DPI cropped tables and images
│   └── processed/              # Serialized JSON MultimodalDocument artifacts
├── scripts/
│   ├── build_test_corpus.py    # Deterministic test corpus generator
│   ├── run_stress_benchmark.py # Multi-threaded performance benchmark runner
│   └── run_citation_audit.py   # 50-element citation audit script
└── tests/
    ├── test_schemas.py          # Geometric invariants & schema validation tests
    ├── test_failure_handling.py # Adversarial edge cases (corrupt, encrypted, empty)
    ├── test_table_integrity.py  # Multi-page continuity & cell value retention
    ├── test_regression_audit.py # Automated 50-element citation regression test
    └── test_ingestion.py        # End-to-end integration tests (PDF, DOCX, PPTX)
```

---

## ⚡ Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Yash990-bit/cogentsearch.git
cd cogentsearch

# Create virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your OpenAI or Anthropic API key
```

### 3. Ingest a Document

```bash
python run_ingestion.py --file data/uploads/demo_architecture.pdf
```

---

## 🧪 Testing & Verification

Run the full automated test suite using a single command:

```bash
pytest tests/ -v
```

```text
============================= test session starts ==============================
collected 19 items

tests/test_failure_handling.py::test_empty_document_rejection PASSED     [  5%]
tests/test_failure_handling.py::test_corrupted_document_rejection PASSED [ 10%]
tests/test_failure_handling.py::test_encrypted_document_rejection PASSED [ 15%]
tests/test_failure_handling.py::test_unsupported_format_rejection PASSED [ 21%]
tests/test_failure_handling.py::test_no_text_pdf_graceful_handling PASSED [ 26%]
tests/test_failure_handling.py::test_duplicate_filename_content_hash_idempotency PASSED [ 31%]
tests/test_ingestion.py::test_pdf_multimodal_ingestion PASSED            [ 36%]
tests/test_ingestion.py::test_docx_ingestion PASSED                      [ 42%]
tests/test_ingestion.py::test_pptx_ingestion PASSED                      [ 47%]
tests/test_ingestion.py::test_complete_multimodal_acceptance_criterion PASSED [ 52%]
tests/test_regression_audit.py::test_citation_integrity_audit_50_elements PASSED [ 57%]
tests/test_schemas.py::test_bounding_box_valid PASSED                    [ 63%]
tests/test_schemas.py::test_bounding_box_invalid_inverted PASSED         [ 68%]
tests/test_schemas.py::test_bounding_box_out_of_bounds PASSED            [ 73%]
tests/test_schemas.py::test_visual_element_contract PASSED               [ 78%]
tests/test_schemas.py::test_multimodal_document_stats PASSED             [ 84%]
tests/test_table_integrity.py::test_multipage_table_continuity_and_ground_truth PASSED [ 89%]
tests/test_table_integrity.py::test_docx_table_ground_truth PASSED       [ 94%]
tests/test_table_integrity.py::test_pptx_table_ground_truth PASSED       [100%]

======================== 19 passed in 1.28s ========================
```

---

## 📊 Stress Test & Benchmark Results

Measured across **22 documents spanning 124 pages/slides** using 5 concurrent workers:

| Metric | Measured Value | Standard Required |
| :--- | :--- | :--- |
| **Total Ingestion Time** | **13.61 seconds** | < 60s |
| **Avg Time per Page** | **0.1098s (109.8ms)** | < 500ms |
| **Peak Memory Footprint** | **61.02 MB** | < 250MB |
| **Ingestion Success Rate** | **100.0% (22/22 docs)** | 100% |
| **Unplanned Failures** | **0** | 0 |
| **Idempotency Match Rate** | **100.0% (22/22 matches)** | 100% |

---

## 🎯 Citation Integrity Audit

Randomly sampled 50 visual elements across the entire test corpus:
- **Result**: **50 / 50 PASSED (100.0%)**
- **Bounding Box Geometry**: 100% strictly within page boundaries `[0, page_dim]`.
- **High-DPI Crops**: 100% exist on disk, non-empty, valid image headers.
- **Provenance Link**: 100% linked to at least one injected retrieval chunk.

See the full auditable log in [CITATION_AUDIT_REPORT.md](CITATION_AUDIT_REPORT.md).

---

## 🔍 Phase 2: Hybrid Indexing & Retrieval Architecture

CogentSearch combines keyword and dense semantic signals with cross-encoder re-ranking:

```
                          User Query
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
        Dense Retrieval               Sparse Retrieval
    (Qdrant Vector Store)               (BM25 Okapi)
               │                             │
               └──────────────┬──────────────┘
                              ▼
                Reciprocal Rank Fusion (RRF, k=60)
                              │
                              ▼ (Top 10 Candidates)
                   Cross-Encoder Reranker
             (ms-marco-MiniLM-L-6-v2 / Scorer)
                              │
                              ▼ (Top 4 Reranked)
                 Final Context Output Chunks
```

### Run Hybrid Search via CLI:

```bash
python run_hybrid_search.py --query "What is the transaction amount and settlement status for TX-1001?"
```

---

## 🗺️ Project Roadmap

- [x] **Phase 1: Advanced Multimodal Ingestion Pipeline** (Docling/PyMuPDF, High-DPI Crop, Vision LLM, Multi-Format Parser, Citation Audit)
- [x] **Phase 2: Hybrid Indexing & Storage Strategy** (Pydantic `IndexedDocument` Schema, Semantic Chunking, Qdrant Vector Store, BM25, RRF Top 10, Cross-Encoder Top 4)
- [ ] **Phase 3: Advanced Reranking Optimization & Context Compression** (Dynamic Context Windows, Metadata Filtering, Token Optimization)
- [ ] **Phase 4: Agentic RAG Loop with LangGraph** (Self-Correction, Hallucination Grader, Query Decomposition, Adaptive Fallbacks)
- [ ] **Phase 5: Automated Evaluation with RAGAS** (Faithfulness, Precision, Recall, Multimodal Ground-Truth Benchmarking)
- [ ] **Phase 6: Interactive Streamlit UI & Visual Citations** (Bounding Box Visualizer & Trace Inspector)

---

## 📄 License

MIT License. Built for enterprise multimodal document intelligence.

