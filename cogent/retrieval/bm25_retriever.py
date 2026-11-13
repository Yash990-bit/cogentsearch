"""
BM25 Keyword Retriever Module for CogentSearch.
Provides persistent lexical and exact keyword matching using BM25Okapi,
with index versioning, disk serialization, deletion/update synchronization,
and Robertson-Lucene non-negative IDF formulation for robust single-document and empty-corpus safety.
"""

from __future__ import annotations
import os
import re
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from cogent.core.config import settings
from cogent.schemas.indexing import IndexedDocument, SearchResultChunk


INDEX_VERSION = "2.0.0"


class BM25Retriever:
    """
    Persistent BM25 Okapi indexer for exact keyword and lexical document retrieval.
    Supports atomic disk persistence, index versioning, deletion synchronization,
    and Robertson-Lucene non-negative IDF computation.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.1,
        persist_dir: Optional[Path | str] = None,
        auto_save: bool = True,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.auto_save = auto_save

        self.documents: List[IndexedDocument] = []
        self._corpus_tokens: List[List[str]] = []
        self._bm25 = None
        self._avg_dl: float = 0.0
        self._doc_lens: List[int] = []
        self._idf: Dict[str, float] = {}

        # Auto-load if persistent directory exists with valid index
        if self.persist_dir and (self.persist_dir / "index_meta.json").exists():
            self.load()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes text into clean lowercase terms, preserving hyphenated terms and alphanumeric IDs."""
        if not text:
            return []
        tokens = re.findall(r"\b[\w$-]+(?:\.[\w$-]+)*\b", text.lower())
        return [t.strip(".,;:?!") for t in tokens if len(t.strip(".,;:?!")) > 1]

    def _compute_corpus_hash(self) -> str:
        """Computes deterministic SHA256 hash of current documents for integrity verification."""
        hasher = hashlib.sha256()
        for doc in self.documents:
            hasher.update(doc.chunk_id.encode("utf-8"))
            hasher.update(doc.content.encode("utf-8"))
        return hasher.hexdigest()

    def index_documents(self, documents: List[IndexedDocument]) -> None:
        """
        Builds or rebuilds BM25 index over documents.
        Safely handles empty document lists without raising ZeroDivisionError.
        """
        if not documents:
            logger.info("Empty document list passed to BM25Retriever; initializing clean empty index.")
            self.documents = []
            self._corpus_tokens = []
            self._bm25 = None
            self._avg_dl = 0.0
            self._doc_lens = []
            self._idf = {}
            if self.auto_save and self.persist_dir:
                self.save()
            return

        self.documents = list(documents)
        self._corpus_tokens = [self.tokenize(doc.content) for doc in self.documents]

        self._build_index()

        logger.info(f"Indexed {len(self.documents)} documents in BM25 retriever.")
        if self.auto_save and self.persist_dir:
            self.save()

    def _build_index(self) -> None:
        """Computes corpus statistics, Robertson-Lucene non-negative IDFs, and BM25Okapi."""
        num_docs = len(self._corpus_tokens)
        if num_docs == 0:
            self._avg_dl = 0.0
            self._doc_lens = []
            self._idf = {}
            self._bm25 = None
            return

        self._doc_lens = [len(t) for t in self._corpus_tokens]
        self._avg_dl = sum(self._doc_lens) / num_docs if num_docs > 0 else 0.0

        # Compute document frequencies
        df: Dict[str, int] = {}
        for doc_tokens in self._corpus_tokens:
            unique_terms = set(doc_tokens)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1

        # Compute Robertson-Lucene non-negative IDF:
        # idf = ln(1.0 + (N - n + 0.5) / (n + 0.5))
        # Guarantees strictly positive IDFs even for single-doc collections (N=1, n=1 -> ln(4/3) > 0)
        self._idf = {}
        for term, freq in df.items():
            base_idf = math.log(1.0 + (num_docs - freq + 0.5) / (freq + 0.5))
            self._idf[term] = max(base_idf, self.epsilon)

        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus_tokens, k1=self.k1, b=self.b)
            # Patch BM25Okapi's negative IDF table with our robust non-negative IDFs
            # rank_bm25 uses self.idf: dict
            if hasattr(self._bm25, "idf"):
                for term, idf_val in self._idf.items():
                    if term in self._bm25.idf:
                        self._bm25.idf[term] = max(self._bm25.idf[term], self.epsilon)
        except ImportError:
            logger.warning("rank-bm25 not installed; using native BM25 implementation.")
            self._bm25 = None

    def search_bm25(
        self,
        query: str,
        top_k: int = 20,
        filter_modality: Optional[str] = None,
        filter_source: Optional[str] = None,
    ) -> List[SearchResultChunk]:
        """Queries BM25 index and returns ranked list of SearchResultChunk objects."""
        if not self.documents:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # If rank_bm25 is available and corpus > 1, use patched scores; otherwise use native scorer
        if self._bm25 is not None and len(self.documents) > 1:
            try:
                raw_scores = self._bm25.get_scores(query_tokens)
            except Exception:
                raw_scores = self._native_score(query_tokens)
        else:
            raw_scores = self._native_score(query_tokens)

        # Pair documents with scores
        scored_pairs = []
        for idx, (doc, score) in enumerate(zip(self.documents, raw_scores)):
            if score <= 0.0:
                continue

            # Apply filters
            if filter_modality and doc.modality_type != filter_modality:
                continue
            if filter_source and doc.source_file != filter_source:
                continue

            scored_pairs.append((doc, float(score)))

        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        results: List[SearchResultChunk] = []
        for rank, (doc, score) in enumerate(scored_pairs[:top_k]):
            results.append(
                SearchResultChunk(
                    chunk_id=doc.chunk_id,
                    content=doc.content,
                    modality_type=doc.modality_type,
                    page_number=doc.page_number,
                    bounding_box=doc.bounding_box,
                    source_file=doc.source_file,
                    headers_hierarchy=doc.headers_hierarchy,
                    bm25_score=round(score, 4),
                    rank=rank + 1,
                    metadata=doc.metadata,
                )
            )

        return results

    def _native_score(self, query_tokens: List[str]) -> List[float]:
        """Computes Robertson-Lucene BM25 scores across all corpus documents."""
        scores = []
        if self._avg_dl == 0.0:
            return [0.0] * len(self.documents)

        for doc_idx, doc_tokens in enumerate(self._corpus_tokens):
            score = 0.0
            doc_len = self._doc_lens[doc_idx]
            term_counts: Dict[str, int] = {}
            for t in doc_tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for q_term in query_tokens:
                if q_term in term_counts:
                    tf = term_counts[q_term]
                    idf = self._idf.get(q_term, self.epsilon)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_dl))
                    score += idf * (numerator / denominator)
            scores.append(score)
        return scores

    # ==========================================================================
    # SYNCHRONIZATION & MUTATION API
    # ==========================================================================

    def delete_chunk(self, chunk_id: str) -> bool:
        """Deletes a chunk by its chunk_id and rebuilds the BM25 index."""
        orig_count = len(self.documents)
        self.documents = [d for d in self.documents if d.chunk_id != chunk_id]
        if len(self.documents) == orig_count:
            return False

        self._corpus_tokens = [self.tokenize(doc.content) for doc in self.documents]
        self._build_index()
        logger.info(f"Deleted chunk '{chunk_id}' from BM25Retriever. Remaining docs: {len(self.documents)}")
        if self.auto_save and self.persist_dir:
            self.save()
        return True

    def delete_by_source(self, source_file: str) -> int:
        """Deletes all chunks originating from source_file and rebuilds the index."""
        orig_count = len(self.documents)
        self.documents = [d for d in self.documents if d.source_file != source_file]
        deleted_count = orig_count - len(self.documents)
        if deleted_count == 0:
            return 0

        self._corpus_tokens = [self.tokenize(doc.content) for doc in self.documents]
        self._build_index()
        logger.info(f"Deleted {deleted_count} chunks for source '{source_file}' from BM25Retriever.")
        if self.auto_save and self.persist_dir:
            self.save()
        return deleted_count

    def update_chunk(self, chunk: IndexedDocument) -> bool:
        """Updates an existing chunk or appends if new, then rebuilds the index."""
        found = False
        for i, existing in enumerate(self.documents):
            if existing.chunk_id == chunk.chunk_id:
                self.documents[i] = chunk
                found = True
                break
        if not found:
            self.documents.append(chunk)

        self._corpus_tokens = [self.tokenize(doc.content) for doc in self.documents]
        self._build_index()
        if self.auto_save and self.persist_dir:
            self.save()
        return True

    def has_chunk(self, chunk_id: str) -> bool:
        """Checks if a chunk_id exists in the BM25 index."""
        return any(d.chunk_id == chunk_id for d in self.documents)

    def count(self) -> int:
        """Returns count of indexed documents."""
        return len(self.documents)

    # ==========================================================================
    # PERSISTENCE & RESTORATION API
    # ==========================================================================

    def save(self, persist_dir: Optional[Path | str] = None) -> bool:
        """
        Atomically saves the tokenized corpus, index metadata, and configuration to disk.
        """
        target_dir = Path(persist_dir or self.persist_dir or (settings.DATA_DIR / "bm25_index"))
        target_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "version": INDEX_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_documents": len(self.documents),
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "corpus_hash": self._compute_corpus_hash(),
        }

        # Serialize documents
        serialized_docs = [doc.model_dump(mode="json") for doc in self.documents]

        meta_path = target_dir / "index_meta.json"
        data_path = target_dir / "corpus_data.json"
        tmp_meta = target_dir / "index_meta.json.tmp"
        tmp_data = target_dir / "corpus_data.json.tmp"

        try:
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            with open(tmp_data, "w", encoding="utf-8") as f:
                json.dump(serialized_docs, f)

            # Atomic replace
            os.replace(tmp_meta, meta_path)
            os.replace(tmp_data, data_path)
            logger.info(f"Saved BM25 index with {len(self.documents)} documents to '{target_dir}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to save BM25 index to '{target_dir}': {e}")
            if tmp_meta.exists():
                tmp_meta.unlink()
            if tmp_data.exists():
                tmp_data.unlink()
            return False

    def load(self, persist_dir: Optional[Path | str] = None) -> bool:
        """
        Loads and restores BM25 index from disk.
        Validates index version and corpus integrity.
        """
        target_dir = Path(persist_dir or self.persist_dir or (settings.DATA_DIR / "bm25_index"))
        meta_path = target_dir / "index_meta.json"
        data_path = target_dir / "corpus_data.json"

        if not meta_path.exists() or not data_path.exists():
            logger.warning(f"No persistent BM25 index files found at '{target_dir}'.")
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Version validation
            saved_ver = meta.get("version")
            if saved_ver != INDEX_VERSION:
                logger.warning(
                    f"Incompatible BM25 index version '{saved_ver}' (expected '{INDEX_VERSION}'). "
                    "Refusing stale index restoration."
                )
                return False

            with open(data_path, "r", encoding="utf-8") as f:
                raw_docs = json.load(f)

            loaded_docs = [IndexedDocument(**d) for d in raw_docs]
            self.documents = loaded_docs
            self.k1 = meta.get("k1", self.k1)
            self.b = meta.get("b", self.b)
            self.epsilon = meta.get("epsilon", self.epsilon)

            # Verify integrity hash
            loaded_hash = self._compute_corpus_hash()
            expected_hash = meta.get("corpus_hash")
            if expected_hash and loaded_hash != expected_hash:
                logger.error("Corrupted BM25 index: corpus hash mismatch. Discarding loaded documents.")
                self.documents = []
                return False

            self._corpus_tokens = [self.tokenize(doc.content) for doc in self.documents]
            self._build_index()

            logger.info(f"Successfully restored BM25 index with {len(self.documents)} documents from '{target_dir}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index from '{target_dir}': {e}")
            self.documents = []
            return False
