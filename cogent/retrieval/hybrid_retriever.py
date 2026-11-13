"""
Hybrid Retriever Module for CogentSearch.
Implements multi-stage hybrid search:
1. Dense Vector Search (Qdrant)
2. Sparse Keyword Search (BM25 with Robertson-Lucene non-negative IDF)
3. Reciprocal Rank Fusion (RRF, k=60)
4. Cross-Encoder Re-Ranking (Top 10 candidates -> Top 4 final relevant chunks).

Includes synchronized deletion, document updates, atomic re-indexing,
index reconciliation between Qdrant and BM25, and public retrieve() interface.
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
    and Cross-Encoder precision reranking with full index synchronization.
    """

    def __init__(
        self,
        qdrant_store: QdrantVectorStore,
        bm25_retriever: BM25Retriever,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        rrf_constant: int = 60,
        auto_reconcile: bool = True,
    ):
        self.qdrant_store = qdrant_store
        self.bm25_retriever = bm25_retriever
        self.cross_encoder = cross_encoder or CrossEncoderReranker(use_local_fallback=False)
        self.rrf_constant = rrf_constant

        # Check for index synchronization on startup
        if auto_reconcile:
            self._check_and_reconcile()

    def _check_and_reconcile(self) -> None:
        """Checks if Qdrant has persisted points while BM25 is empty; reconciles if needed."""
        try:
            qdrant_count = self.qdrant_store.count()
            bm25_count = self.bm25_retriever.count()

            if qdrant_count > 0 and bm25_count == 0:
                logger.warning(
                    f"Index desynchronization detected on startup: Qdrant has {qdrant_count} points, "
                    f"but BM25 has {bm25_count} documents. Automatically reconciling BM25 from Qdrant payloads..."
                )
                self.reconcile_from_vectorstore()
        except Exception as e:
            logger.error(f"Error during index startup reconciliation check: {e}")

    def index_documents(self, documents: List[IndexedDocument]) -> None:
        """Indexes documents into both Qdrant (dense vectors) and BM25 (lexical index)."""
        logger.info(f"Indexing {len(documents)} documents into HybridRetriever...")
        self.qdrant_store.upsert_documents(documents)
        self.bm25_retriever.index_documents(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        filter_modality: Optional[str] = None,
        filter_source: Optional[str] = None,
    ) -> List[SearchResultChunk]:
        """
        Public standard retrieval interface for CogentSearch.
        Executes the full hybrid retrieval pipeline (Dense + BM25 -> RRF -> Cross-Encoder).
        """
        return self.search(
            query=query,
            final_top_k=top_k,
            filter_modality=filter_modality,
            filter_source=filter_source,
        )

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

        if not query or not query.strip():
            logger.warning("Empty query received in hybrid search; returning empty list.")
            return []

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

        # 4. Extract Top candidates for Cross-Encoder re-ranking
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

        # 6. Sort candidates descending by Cross-Encoder score
        top_candidates.sort(key=lambda c: (c.cross_encoder_score or 0.0), reverse=True)

        # 7. Output Top final chunks
        final_top = top_candidates[:final_top_k]
        for final_rank, chunk in enumerate(final_top, start=1):
            chunk.rank = final_rank

        logger.info(f"Completed Cross-Encoder reranking. Selected Top {len(final_top)} chunks.")
        return final_top

    # ==========================================================================
    # SYNCHRONIZED MUTATION & RECONCILIATION API
    # ==========================================================================

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Synchronously deletes a chunk from both Qdrant and BM25 indexes.
        Returns True if deleted from at least one index.
        """
        q_ok = self.qdrant_store.delete_chunk(chunk_id)
        b_ok = self.bm25_retriever.delete_chunk(chunk_id)
        logger.info(f"Synchronized delete_chunk('{chunk_id}'): Qdrant={q_ok}, BM25={b_ok}")
        return q_ok or b_ok

    def delete_by_source(self, source_file: str) -> int:
        """
        Synchronously deletes all chunks belonging to source_file from both indexes.
        Returns the count of chunks deleted from Qdrant.
        """
        q_count = self.qdrant_store.delete_by_source(source_file)
        b_count = self.bm25_retriever.delete_by_source(source_file)
        logger.info(f"Synchronized delete_by_source('{source_file}'): Qdrant={q_count}, BM25={b_count}")
        return q_count

    def update_document(self, documents: List[IndexedDocument], source_file: Optional[str] = None) -> int:
        """
        Synchronously updates or re-indexes chunks for a document.
        Removes existing chunks for the source file and indexes new ones atomically.
        """
        target_source = source_file or (documents[0].source_file if documents else None)
        if target_source:
            self.delete_by_source(target_source)

        if documents:
            self.index_documents(documents)
        return len(documents)

    def reindex_document(self, documents: List[IndexedDocument], source_file: str) -> int:
        """Explicit alias for update_document with mandatory source_file parameter."""
        return self.update_document(documents, source_file=source_file)

    def reconcile_from_vectorstore(self) -> int:
        """
        Reconstructs the BM25 index from persistent Qdrant points.
        Guarantees complete index synchronization across restarts or after partial failures.
        """
        all_chunks = self.qdrant_store.get_all_chunks()
        indexed_docs = [
            IndexedDocument(
                chunk_id=c.chunk_id,
                content=c.content,
                modality_type=c.modality_type,
                page_number=c.page_number,
                bounding_box=c.bounding_box,
                source_file=c.source_file,
                headers_hierarchy=c.headers_hierarchy,
                metadata=c.metadata,
            )
            for c in all_chunks
        ]
        self.bm25_retriever.index_documents(indexed_docs)
        logger.info(f"Reconciled BM25 index from Qdrant: {len(indexed_docs)} documents synchronized.")
        return len(indexed_docs)
