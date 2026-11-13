"""
Local Persistent Qdrant Vector Store Connector for CogentSearch.
Supports local directory persistence (./data/qdrant_db) and in-memory test mode,
with cosine distance indexing, metadata payload filtering, and bulk upsert operations.
"""

from __future__ import annotations
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from cogent.core.config import settings
from cogent.core.exceptions import VectorSpaceMismatchError
from cogent.schemas.indexing import IndexedDocument, SearchResultChunk
from cogent.schemas.document import BoundingBox
from cogent.vectorstore.embeddings import EmbeddingEngine


class QdrantVectorStore:
    """Manages local Qdrant collections for multimodal vector indexing and similarity retrieval."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[Path | str] = None,
        in_memory: bool = False,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        self.collection_name = collection_name or "cogent_search_documents"
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.in_memory = in_memory

        if in_memory:
            self.persist_path = ":memory:"
        else:
            self.persist_dir = Path(persist_dir or (settings.DATA_DIR / "qdrant_db"))
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.persist_path = str(self.persist_dir)

        self._client = None
        self._init_client()

    def _init_client(self):
        """Initializes local QdrantClient instance and ensures collection exists."""
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        if self.in_memory:
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(path=self.persist_path)

        # Check if collection exists
        collections = [c.name for c in self._client.get_collections().collections]
        meta_file = (self.persist_dir / f"{self.collection_name}_meta.json") if not self.in_memory else None

        if self.collection_name not in collections:
            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' "
                f"with vector dimension {self.embedding_engine.dimension} (Cosine)."
            )
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_engine.dimension,
                    distance=Distance.COSINE,
                ),
            )
            if meta_file is not None:
                meta_payload = {
                    "collection_name": self.collection_name,
                    "provider": self.embedding_engine.provider,
                    "model_name": self.embedding_engine.model_name,
                    "dimension": self.embedding_engine.dimension,
                    "distance": "Cosine",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta_payload, f, indent=2)
        else:
            # Collection exists: verify embedding space compatibility
            if meta_file is not None and meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        saved_meta = json.load(f)
                    
                    saved_provider = saved_meta.get("provider")
                    saved_model = saved_meta.get("model_name")
                    curr_provider = self.embedding_engine.provider
                    curr_model = self.embedding_engine.model_name

                    if saved_provider != curr_provider or saved_model != curr_model:
                        raise VectorSpaceMismatchError(
                            f"Vector collection '{self.collection_name}' was created with provider '{saved_provider}' "
                            f"and model '{saved_model}', but current runtime is configured with provider '{curr_provider}' "
                            f"and model '{curr_model}'. Mixing vectors from different embedding spaces produces corrupted similarity rankings."
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Could not validate collection metadata file {meta_file}: {e}")

    def upsert_documents(self, documents: List[IndexedDocument]) -> int:
        """
        Embeds and upserts a list of IndexedDocument instances into the local Qdrant collection.
        Returns the count of points upserted.
        """
        if not documents:
            return 0

        from qdrant_client.http.models import PointStruct

        texts = [doc.content for doc in documents]
        vectors = self.embedding_engine.embed_documents(texts)

        points: List[PointStruct] = []
        for doc, vector in zip(documents, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.chunk_id))
            bbox_dict = doc.bounding_box.model_dump() if doc.bounding_box else None

            payload = {
                "chunk_id": doc.chunk_id,
                "content": doc.content,
                "modality_type": doc.modality_type,
                "page_number": doc.page_number,
                "bounding_box": bbox_dict,
                "source_file": doc.source_file,
                "headers_hierarchy": doc.headers_hierarchy,
                "metadata": doc.metadata,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(f"Upserted {len(points)} documents into Qdrant collection '{self.collection_name}'.")
        return len(points)

    def search_dense(
        self,
        query: str,
        top_k: int = 20,
        filter_modality: Optional[str] = None,
        filter_source: Optional[str] = None,
    ) -> List[SearchResultChunk]:
        """Performs dense vector similarity search against Qdrant."""
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        query_vector = self.embedding_engine.embed_query(query)

        # Build optional Qdrant payload filters
        conditions = []
        if filter_modality:
            conditions.append(FieldCondition(key="modality_type", match=MatchValue(value=filter_modality)))
        if filter_source:
            conditions.append(FieldCondition(key="source_file", match=MatchValue(value=filter_source)))

        query_filter = Filter(must=conditions) if conditions else None

        # Modern Qdrant client 1.10+ query_points API
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            scored_points = response.points
        else:
            scored_points = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

        results: List[SearchResultChunk] = []
        for idx, pt in enumerate(scored_points):
            payload = pt.payload or {}
            bbox_raw = payload.get("bounding_box")
            bbox = BoundingBox(**bbox_raw) if bbox_raw else None

            results.append(
                SearchResultChunk(
                    chunk_id=payload.get("chunk_id", str(pt.id)),
                    content=payload.get("content", ""),
                    modality_type=payload.get("modality_type", "text"),
                    page_number=payload.get("page_number", 1),
                    bounding_box=bbox,
                    source_file=payload.get("source_file", ""),
                    headers_hierarchy=payload.get("headers_hierarchy", []),
                    dense_score=round(float(pt.score), 4),
                    rank=idx + 1,
                    metadata=payload.get("metadata", {}),
                )
            )

        return results

    def count(self) -> int:
        """Returns total point count in collection."""
        res = self._client.count(collection_name=self.collection_name)
        return res.count

    def has_chunk(self, chunk_id: str) -> bool:
        """Checks if a point with given chunk_id exists in Qdrant."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        try:
            pts = self._client.retrieve(collection_name=self.collection_name, ids=[point_id])
            return len(pts) > 0
        except Exception:
            return False

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Deletes a specific chunk by its chunk_id from Qdrant.
        Returns True if chunk was present and deleted, False otherwise.
        """
        from qdrant_client.http.models import PointIdsList

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        if not self.has_chunk(chunk_id):
            return False

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[point_id]),
            wait=True,
        )
        logger.info(f"Deleted chunk '{chunk_id}' (point {point_id}) from Qdrant collection '{self.collection_name}'.")
        return True

    def delete_by_source(self, source_file: str) -> int:
        """
        Deletes all chunks originating from source_file in Qdrant.
        Returns the count of points removed.
        """
        from qdrant_client.http.models import FilterSelector, Filter, FieldCondition, MatchValue

        count_before = self.count()
        query_filter = Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))])

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=query_filter),
            wait=True,
        )
        count_after = self.count()
        deleted_count = max(0, count_before - count_after)
        logger.info(f"Deleted {deleted_count} chunks for source '{source_file}' from Qdrant.")
        return deleted_count

    def get_all_chunks(self) -> List[SearchResultChunk]:
        """
        Scrolls and retrieves all indexed points with payloads from Qdrant.
        Used for index reconciliation and BM25 index reconstruction.
        """
        results: List[SearchResultChunk] = []
        offset = None

        while True:
            records, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in records:
                payload = pt.payload or {}
                bbox_raw = payload.get("bounding_box")
                bbox = BoundingBox(**bbox_raw) if bbox_raw else None

                results.append(
                    SearchResultChunk(
                        chunk_id=payload.get("chunk_id", str(pt.id)),
                        content=payload.get("content", ""),
                        modality_type=payload.get("modality_type", "text"),
                        page_number=payload.get("page_number", 1),
                        bounding_box=bbox,
                        source_file=payload.get("source_file", ""),
                        headers_hierarchy=payload.get("headers_hierarchy", []),
                        metadata=payload.get("metadata", {}),
                    )
                )

            if next_offset is None or len(records) == 0:
                break
            offset = next_offset

        return results

