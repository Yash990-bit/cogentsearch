"""
LangGraph StateGraph Workflow for CogentSearch Phase 3.
Assembles the complete self-correcting agentic RAG loop:
Query Rewriter -> Retriever -> Document Grader -> (Conditional Loop / Web Search) -> Generator (Answer & Cite).
"""

from __future__ import annotations
from typing import Literal, Optional, Any
from loguru import logger
from langgraph.graph import StateGraph, END

from cogent.agent.state import AgentState
from cogent.agent.nodes import AgentNodes
from cogent.retrieval.hybrid_retriever import HybridRetriever


def route_after_grading(state: AgentState) -> Literal["generator", "query_rewriter", "web_search"]:
    """
    Conditional routing edge after document grading:
    1. If documents contain the answer -> proceed to 'generator'.
    2. If documents are insufficient and retry allowance remains -> loop back to 'query_rewriter'.
    3. If retries are exhausted -> fallback to 'web_search'.
    """
    is_relevant = state.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if is_relevant:
        logger.info("Grader routed to: 'generator' (documents contain the answer).")
        return "generator"

    if retry_count < max_retries:
        logger.info(f"Grader routed back to: 'query_rewriter' (attempt {retry_count + 1}/{max_retries}).")
        return "query_rewriter"

    logger.info("Grader routed to: 'web_search' (retries exhausted, falling back to web).")
    return "web_search"


def build_agentic_rag_graph(
    hybrid_retriever: HybridRetriever,
    llm: Optional[Any] = None,
    max_retries: int = 2,
) -> Any:
    """
    Constructs and compiles the LangGraph StateGraph agent workflow.
    """
    nodes = AgentNodes(
        hybrid_retriever=hybrid_retriever,
        llm=llm,
        max_retries=max_retries,
    )

    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("query_rewriter", nodes.query_rewriter)
    workflow.add_node("retriever", nodes.retriever_node)
    workflow.add_node("doc_grader", nodes.doc_grader)
    workflow.add_node("web_search", nodes.web_search_node)
    workflow.add_node("generator", nodes.generator_node)

    # Set Entry Point
    workflow.set_entry_point("query_rewriter")

    # Fixed Edges
    workflow.add_edge("query_rewriter", "retriever")
    workflow.add_edge("retriever", "doc_grader")

    # Conditional Routing from Document Grader
    workflow.add_conditional_edges(
        "doc_grader",
        route_after_grading,
        {
            "generator": "generator",
            "query_rewriter": "query_rewriter",
            "web_search": "web_search",
        },
    )

    # Edge from Web Search to Generator
    workflow.add_edge("web_search", "generator")

    # Final Edge from Generator to END
    workflow.add_edge("generator", END)

    compiled_app = workflow.compile()
    logger.info("Successfully compiled LangGraph Agentic RAG StateGraph.")
    return compiled_app

