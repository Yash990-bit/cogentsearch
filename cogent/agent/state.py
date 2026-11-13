"""
Agent State Definition for CogentSearch Phase 3 LangGraph Loop.
Represents the complete runtime memory of the self-correcting agentic RAG pipeline.
"""

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field

from cogent.schemas.indexing import SearchResultChunk


class AgentState(TypedDict):
    """Runtime state dictionary passed between LangGraph nodes."""

    # User input query
    question: str

    # Rewritten/optimized search query
    rewritten_query: str

    # Chunks retrieved from Phase 2 Hybrid Search
    documents: List[SearchResultChunk]

    # Rewrite retry counter and maximum allowance before web fallback
    retry_count: int
    max_retries: int

    # Document grading judgment
    is_relevant: bool
    grade_reason: str

    # Web search fallback status and documents
    web_search_needed: bool
    web_documents: List[Dict[str, Any]]

    # Generated final answer
    generation: str

    # Extracted structured citations
    citations: List[Dict[str, Any]]

    # Audit execution trace of graph node traversal
    trace: List[str]


class GradeResult(BaseModel):
    """Structured output schema for Document Grader LLM node."""

    binary_score: str = Field(
        description="Whether the retrieved chunks contain information relevant to answering the question: 'yes' or 'no'"
    )
    reasoning: str = Field(
        description="Explanation of why the chunks are or are not sufficient to answer the question"
    )


class RewriteResult(BaseModel):
    """Structured output schema for Query Rewriter LLM node."""

    optimized_query: str = Field(
        description="Search query optimized for dense semantic retrieval and keyword matching"
    )
    explanation: str = Field(
        description="Brief explanation of how the query was refined"
    )


class CitationItem(BaseModel):
    """Structured citation item linking claims to source document metadata."""

    source_file: str
    page_number: int
    modality_type: str
    citation_tag: str
    bounding_box: Optional[Dict[str, Any]] = None
