"""
Agentic RAG Engine for CogentSearch.
Provides a unified high-level interface to execute the compiled LangGraph loop.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from loguru import logger
from pydantic import BaseModel, Field

from cogent.schemas.indexing import SearchResultChunk
from cogent.agent.graph import build_agentic_rag_graph
from cogent.agent.state import AgentState, CitationItem
from cogent.retrieval.hybrid_retriever import HybridRetriever
from cogent.vectorstore.qdrant_store import QdrantVectorStore
from cogent.retrieval.bm25_retriever import BM25Retriever
from cogent.core.config import settings


class AgenticResponse(BaseModel):
    """Structured response returned by the Agentic RAG loop."""

    question: str
    rewritten_query: str
    generation: str
    citations: List[CitationItem]
    retrieved_chunk_count: int
    retrieved_documents: List[SearchResultChunk] = Field(default_factory=list)
    web_search_used: bool
    retry_count: int
    trace: List[str]


class AgenticRAG:
    """
    High-level orchestrator for the LangGraph Agentic RAG pipeline.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        llm: Optional[Any] = None,
        max_retries: int = 2,
    ):
        if hybrid_retriever is None:
            logger.info("Initializing default HybridRetriever for AgenticRAG...")
            qdrant = QdrantVectorStore()
            bm25 = BM25Retriever()
            hybrid_retriever = HybridRetriever(qdrant_store=qdrant, bm25_retriever=bm25)

        self.hybrid_retriever = hybrid_retriever
        self.max_retries = max_retries
        self.app = build_agentic_rag_graph(
            hybrid_retriever=self.hybrid_retriever,
            llm=llm,
            max_retries=max_retries,
        )

    def run(self, question: str) -> AgenticResponse:
        """
        Executes the LangGraph loop for a given user question and returns an AgenticResponse.
        """
        initial_state: AgentState = {
            "question": question,
            "rewritten_query": "",
            "documents": [],
            "retry_count": 0,
            "max_retries": self.max_retries,
            "is_relevant": False,
            "grade_reason": "",
            "web_search_needed": False,
            "web_documents": [],
            "generation": "",
            "citations": [],
            "trace": [],
        }

        logger.info(f"Invoking AgenticRAG LangGraph loop for question: '{question}'")
        final_state: AgentState = self.app.invoke(initial_state)

        citations = [CitationItem(**c) for c in final_state.get("citations", [])]

        return AgenticResponse(
            question=final_state["question"],
            rewritten_query=final_state.get("rewritten_query", ""),
            generation=final_state.get("generation", ""),
            citations=citations,
            retrieved_chunk_count=len(final_state.get("documents", [])),
            retrieved_documents=final_state.get("documents", []),
            web_search_used=final_state.get("web_search_needed", False),
            retry_count=final_state.get("retry_count", 0),
            trace=final_state.get("trace", []),
        )

