"""
Hybrid Retriever Module for CogentSearch.
Implements multi-stage hybrid search:
1. Dense Vector Search (Qdrant)
2. Sparse Keyword Search (BM25)
3. Reciprocal Rank Fusion (RRF, k=60)
4. Cross-Encoder Re-Ranking (Top 10 candidates -> Top 4 final relevant chunks).
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from loguru import logger

from cogent.schemas.indexing import IndexedDocument, SearchResultChunk
from cogent.vectorstore.qdrant_store import QdrantVectorStore
from cogent.retrieval.bm25_retriever import BM25Retriever
from cogent.retrieval.cross_encoder import CrossEncoderReranker


class HybridRetriever:
    """
    Orchestrates Dense + BM25 Hybrid Retrieval, Reciprocal Rank Fusion (RRF),
    and Cross-Encoder precision reranking.
    """

    def __init__(
        self,
        qdrant_store: QdrantVectorStore,
        bm25_retriever: BM25Retriever,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        rrf_constant: int = 60,
    ):
        self.qdrant_store = qdrant_store
        self.bm25_retriever = bm25_retriever
        self.cross_encoder = cross_encoder or CrossEncoderReranker(use_local_fallback=False)
        self.rrf_constant = rrf_constant

    def index_documents(self, documents: List[IndexedDocument]) -> None:
        """Indexes documents into both Qdrant (dense vectors) and BM25 (lexical index)."""
        logger.info(f"Indexing {len(documents)} documents into HybridRetriever...")
        self.qdrant_store.upsert_documents(documents)
        self.bm25_retriever.index_documents(documents)

    def search(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        top_n_rerank: int = 10,
        final_top_k: int = 4,
        filter_modality: Optional[str] = None,
        filter_source: Optional[str] = None,
    ) -> List[SearchResultChunk]:
        """
        Executes the full hybrid retrieval pipeline:
        1. Parallel dense vector search & BM25 keyword search.
        2. Merges ranked candidate lists using Reciprocal Rank Fusion (RRF).
        3. Takes Top 10 RRF candidates.
        4. Cross-Encoder reranks the 10 candidates to yield the final Top 4 chunks.
        """
        logger.info(f"Hybrid retrieval query: '{query}'")

        # 1. Dense Vector Retrieval
        dense_results = self.qdrant_store.search_dense(
            query=query,
            top_k=top_k_dense,
            filter_modality=filter_modality,
            filter_source=filter_source,
        )

        # 2. BM25 Keyword Retrieval
        bm25_results = self.bm25_retriever.search_bm25(
            query=query,
            top_k=top_k_bm25,
            filter_modality=filter_modality,
            filter_source=filter_source,
        )

        # 3. Reciprocal Rank Fusion (RRF)
        # Score(d) = sum(1 / (k + rank))
        rrf_map: Dict[str, Dict[str, Any]] = {}

        # Process Dense Ranks
        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk_id
            if cid not in rrf_map:
                rrf_map[cid] = {
                    "chunk": res,
                    "rrf_score": 0.0,
                    "dense_score": res.dense_score,
                    "bm25_score": None,
                }
            rrf_map[cid]["rrf_score"] += 1.0 / (self.rrf_constant + rank)
            rrf_map[cid]["dense_score"] = res.dense_score

        # Process BM25 Ranks
        for rank, res in enumerate(bm25_results, start=1):
            cid = res.chunk_id
            if cid not in rrf_map:
                rrf_map[cid] = {
                    "chunk": res,
                    "rrf_score": 0.0,
                    "dense_score": None,
                    "bm25_score": res.bm25_score,
                }
            rrf_map[cid]["rrf_score"] += 1.0 / (self.rrf_constant + rank)
            rrf_map[cid]["bm25_score"] = res.bm25_score

        if not rrf_map:
            logger.warning("No candidates retrieved from either Dense or BM25 searches.")
            return []

        # Sort all unique candidates descending by RRF score
        rrf_sorted = sorted(rrf_map.values(), key=lambda x: x["rrf_score"], reverse=True)

        # 4. Extract Top 10 candidates for Cross-Encoder re-ranking
        top_candidates_meta = rrf_sorted[:top_n_rerank]
        top_candidates = []
        for item in top_candidates_meta:
            chunk = item["chunk"]
            chunk.rrf_score = round(item["rrf_score"], 6)
            chunk.dense_score = item["dense_score"]
            chunk.bm25_score = item["bm25_score"]
            top_candidates.append(chunk)

        logger.info(
            f"RRF fused {len(rrf_map)} unique candidates -> selected Top {len(top_candidates)} "
            "for Cross-Encoder reranking."
        )

        # 5. Cross-Encoder Re-Ranking
        pairs = [(query, c.content) for c in top_candidates]
        ce_scores = self.cross_encoder.predict(pairs)

        for chunk, ce_score in zip(top_candidates, ce_scores):
            chunk.cross_encoder_score = ce_score

        # 6. Sort Top 10 candidates descending by Cross-Encoder score
        top_candidates.sort(key=lambda c: (c.cross_encoder_score or 0.0), reverse=True)

        # 7. Output Top 4 most relevant chunks
        final_top = top_candidates[:final_top_k]
        for final_rank, chunk in enumerate(final_top, start=1):
            chunk.rank = final_rank

        logger.info(f"Completed Cross-Encoder reranking. Selected Top {len(final_top)} chunks.")
        return final_top

