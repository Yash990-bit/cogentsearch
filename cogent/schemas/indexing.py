"""
Phase 2 Indexing and Retrieval Schemas for CogentSearch.
Defines Pydantic V2 models for indexed document chunks, hybrid search results,
and cross-encoder reranked outputs with full multimodal provenance.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict

from cogent.schemas.document import BoundingBox


class IndexedDocument(BaseModel):
    """
    Standard Document Schema for Vector Indexing & Hybrid Retrieval.
    Explicitly holds:
    - content: Raw or enriched text chunk content
    - modality_type: Categorization ('text', 'table', 'image')
    - page_number: 1-indexed document page/slide
    - bounding_box: Coordinates of the element on the page (or None for full-page text)
    - source_file: Source document name / file path
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    chunk_id: str = Field(..., description="Unique deterministic identifier for vector indexing")
    content: str = Field(..., description="Text content used for embedding and keyword search")
    modality_type: Literal["text", "table", "image"] = Field(
        default="text",
        description="Modality of the chunk: 'text', 'table', or 'image'"
    )
    page_number: int = Field(..., ge=1, description="1-indexed source page or slide number")
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Coordinates of the visual or tabular asset on the page, if applicable"
    )
    source_file: str = Field(..., description="Filename or relative path of the source document")
    headers_hierarchy: List[str] = Field(default_factory=list, description="Section heading hierarchy")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata for filtering/reranking")


class SearchResultChunk(BaseModel):
    """
    Represents a retrieved document chunk after Hybrid Search (Dense + BM25 + RRF)
    and Cross-Encoder Re-Ranking.
    """
    chunk_id: str
    content: str
    modality_type: Literal["text", "table", "image"]
    page_number: int
    bounding_box: Optional[BoundingBox] = None
    source_file: str
    headers_hierarchy: List[str] = Field(default_factory=list)
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    rank: int = Field(default=1, description="Final rank after reranking")
    metadata: Dict[str, Any] = Field(default_factory=dict)

