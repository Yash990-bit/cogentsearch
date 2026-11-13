"""
BM25 Keyword Retriever Module for CogentSearch.
Provides lexical and exact keyword matching using BM25Okapi,
enabling precise retrieval of alphanumeric IDs, acronyms, and technical terms.
"""

from __future__ import annotations
import re
import math
from typing import List, Dict, Any, Optional
from loguru import logger

from cogent.schemas.indexing import IndexedDocument, SearchResultChunk


class BM25Retriever:
    """In-memory BM25 Okapi indexer for exact keyword and lexical document retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[IndexedDocument] = []
        self._corpus_tokens: List[List[str]] = []
        self._bm25 = None
        self._avg_dl: float = 0.0
        self._doc_lens: List[int] = []
        self._idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes text into clean lowercase terms, preserving hyphenated terms and alphanumeric IDs."""
        # Match words, hyphenated technical terms, numbers, and currency values
        tokens = re.findall(r"\b[\w$-]+(?:\.[\w$-]+)*\b", text.lower())
        return [t.strip(".,;:?!") for t in tokens if len(t.strip(".,;:?!")) > 1]

    def index_documents(self, documents: List[IndexedDocument]) -> None:
        """Builds BM25 index over documents."""
        self.documents = list(documents)
        self._corpus_tokens = [self.tokenize(doc.content) for doc in self.documents]

        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus_tokens, k1=self.k1, b=self.b)
        except ImportError:
            logger.warning("rank-bm25 not installed; falling back to native BM25 implementation.")
            self._build_native_bm25()

        logger.info(f"Indexed {len(self.documents)} documents in BM25 retriever.")

    def _build_native_bm25(self) -> None:
        """Lightweight native BM25 implementation fallback."""
        num_docs = len(self._corpus_tokens)
        self._doc_lens = [len(t) for t in self._corpus_tokens]
        self._avg_dl = sum(self._doc_lens) / num_docs if num_docs > 0 else 0.0

        # Compute document frequencies
        df: Dict[str, int] = {}
        for doc_tokens in self._corpus_tokens:
            unique_terms = set(doc_tokens)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1

        # Compute IDF
        self._idf = {}
        for term, freq in df.items():
            self._idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

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

        if self._bm25:
            scores = self._bm25.get_scores(query_tokens)
        else:
            scores = self._native_score(query_tokens)

        # Pair documents with scores
        scored_pairs = []
        for idx, (doc, score) in enumerate(zip(self.documents, scores)):
            if score <= 0:
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
        """Computes native BM25 scores across all corpus documents."""
        scores = []
        for doc_idx, doc_tokens in enumerate(self._corpus_tokens):
            score = 0.0
            doc_len = self._doc_lens[doc_idx]
            term_counts = {}
            for t in doc_tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for q_term in query_tokens:
                if q_term in term_counts:
                    tf = term_counts[q_term]
                    idf = self._idf.get(q_term, 0.0)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_dl))
                    score += idf * (numerator / denominator)
            scores.append(score)
        return scores

