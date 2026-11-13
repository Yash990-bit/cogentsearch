"""
LangGraph Nodes for CogentSearch Phase 3 Agentic Loop.
Implements:
1. QueryRewriterNode: Optimizes raw user queries for hybrid vector & lexical search.
2. RetrieverNode: Invokes Phase 2 HybridRetriever (Dense + BM25 -> RRF -> Cross-Encoder).
3. DocumentGraderNode: Evaluates chunk relevance and factual answer presence (Self-Correction).
4. WebSearchNode: Fallback external search when internal corpus lacks the required knowledge.
5. GeneratorNode: Synthesizes final answer with strict citation syntax [Source: File, Page X, Type: Y].
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from cogent.core.config import settings
from cogent.schemas.indexing import SearchResultChunk
from cogent.retrieval.hybrid_retriever import HybridRetriever
from cogent.agent.state import AgentState, GradeResult, RewriteResult, CitationItem
from cogent.agent.compressor import ContextCompressor
from cogent.agent.prompts import (
    REWRITER_SYSTEM_PROMPT,
    GRADER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
)


class AgentNodes:
    """Encapsulates LangGraph node implementations for CogentSearch."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        llm: Optional[Any] = None,
        max_retries: int = 2,
    ):
        self.retriever = hybrid_retriever
        self.max_retries = max_retries
        self._llm = llm
        self._init_llm()

    def _init_llm(self) -> None:
        """Initializes LangChain ChatOpenAI if API key exists, otherwise activates local fallback."""
        if self._llm is not None:
            return

        api_key = settings.OPENAI_API_KEY
        if api_key:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=settings.LLM_MODEL_NAME,
                    temperature=settings.LLM_TEMPERATURE,
                    api_key=api_key,
                )
                logger.info(f"Initialized ChatOpenAI model '{settings.LLM_MODEL_NAME}'.")
            except Exception as e:
                logger.warning(f"Could not initialize ChatOpenAI: {e}. Using deterministic agent fallback.")
                self._llm = None
        else:
            logger.info("Operating in deterministic offline Agent LLM mode.")
            self._llm = None

    # =========================================================================
    # Node 1: Query Rewriter
    # =========================================================================
    def query_rewriter(self, state: AgentState) -> Dict[str, Any]:
        """
        Rewrites the raw user query into an optimized query for dense vector
        and BM25 keyword search. Incorporates grader feedback if on a retry loop.
        """
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        grade_reason = state.get("grade_reason", "")
        trace = list(state.get("trace", []))
        trace.append(f"query_rewriter (attempt {retry_count + 1})")

        logger.info(f"Executing Query Rewriter (attempt {retry_count + 1}) for: '{question}'")

        if self._llm is not None:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                feedback_prompt = ""
                if retry_count > 0 and grade_reason:
                    feedback_prompt = f"\nPrevious retrieval attempt failed because: {grade_reason}. Please reformulate with alternate domain keywords or broader synonyms."

                messages = [
                    SystemMessage(content=REWRITER_SYSTEM_PROMPT),
                    HumanMessage(content=f"Original Query: {question}{feedback_prompt}\nOptimized Search Query:"),
                ]
                response = self._llm.invoke(messages)
                rewritten = response.content.strip().replace('"', "")
            except Exception as e:
                logger.error(f"Error in LLM query rewrite: {e}. Using rule-based rewrite.")
                rewritten = self._rule_based_rewrite(question, retry_count)
        else:
            rewritten = self._rule_based_rewrite(question, retry_count)

        logger.info(f"Rewritten Query: '{rewritten}'")
        return {
            "rewritten_query": rewritten,
            "retry_count": retry_count + 1,
            "trace": trace,
        }

    @staticmethod
    def _rule_based_rewrite(query: str, retry_count: int) -> str:
        """Deterministic query optimizer stripping conversational noise and expanding terms."""
        cleaned = re.sub(
            r"^(can\s+you\s+|could\s+you\s+|would\s+you\s+)?(please\s+)?(find|tell\s+me(\s+about)?|search\s+for|look\s+up|what\s+is|what\s+are|i\s+want\s+to\s+know(\s+what\s+(is|are))?|i\s+need\s+to\s+(find|know)|show\s+me|give\s+me)\s+",
            "",
            query.strip(),
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[?!.,;]+$", "", cleaned).strip()

        # On retries, expand with domain synonyms
        if retry_count > 0:
            cleaned = f"{cleaned} specifications parameters details overview"
        return cleaned or query

    # =========================================================================
    # Node 2: Retriever
    # =========================================================================
    def retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes Phase 2 Hybrid Search (Dense + BM25 -> RRF -> Cross-Encoder).
        """
        query_to_search = state.get("rewritten_query") or state["question"]
        trace = list(state.get("trace", []))
        trace.append("retriever")

        logger.info(f"Executing Hybrid Retriever for query: '{query_to_search}'")
        retrieved_chunks = self.retriever.retrieve(query=query_to_search, top_k=4)
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks from Phase 2 hybrid store.")

        return {
            "documents": retrieved_chunks,
            "trace": trace,
        }

    # =========================================================================
    # Node 3: Document Grader (Self-Correction)
    # =========================================================================
    def doc_grader(self, state: AgentState) -> Dict[str, Any]:
        """
        Grades whether the retrieved chunks actually contain the factual answer.
        Returns is_relevant ('yes'/'no') and grade_reason.
        """
        question = state["question"]
        documents = state.get("documents", [])
        trace = list(state.get("trace", []))
        trace.append("doc_grader")

        logger.info(f"Grading {len(documents)} retrieved documents against query: '{question}'")

        if not documents:
            logger.warning("No documents retrieved; grading as non-relevant.")
            return {
                "is_relevant": False,
                "grade_reason": "No documents were retrieved from the local index.",
                "trace": trace,
            }

        context_str = "\n\n".join(
            f"--- Document [Source: {d.source_file}, Page: {d.page_number}, Type: {d.modality_type}] ---\n{d.content}"
            for d in documents
        )

        if self._llm is not None:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=GRADER_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"User Question: {question}\n\n"
                            f"Retrieved Documents:\n{context_str}\n\n"
                            "Assess relevance. Reply in format:\n"
                            "BINARY_SCORE: <yes/no>\n"
                            "REASONING: <explanation>"
                        )
                    ),
                ]
                response = self._llm.invoke(messages).content.strip()
                score_match = re.search(r"BINARY_SCORE:\s*(yes|no)", response, re.IGNORECASE)
                is_relevant = (score_match.group(1).lower() == "yes") if score_match else ("yes" in response.lower()[:30])
                reasoning = response
            except Exception as e:
                logger.error(f"Error in LLM document grading: {e}. Using deterministic grader.")
                is_relevant, reasoning = self._deterministic_grade(question, documents)
        else:
            is_relevant, reasoning = self._deterministic_grade(question, documents)

        logger.info(f"Document Grader Verdict: is_relevant={is_relevant} | Reason: {reasoning[:120]}...")
        return {
            "is_relevant": is_relevant,
            "grade_reason": reasoning,
            "trace": trace,
        }

    DOMAIN_SYNONYMS = {
        "heart attack": ["myocardial infarction", "cardiac arrest", "coronary event"],
        "myocardial infarction": ["heart attack", "cardiac infarction"],
        "myocardial infarctions": ["heart attack", "heart attacks"],
        "frequency": ["incidence", "rate", "prevalence", "occurrence"],
        "incidence": ["frequency", "rate", "occurrence"],
        "study": ["evaluation", "trial", "investigation", "analysis"],
        "evaluation": ["study", "trial", "assessment"],
        "revenue": ["turnover", "sales", "receipts", "gross income"],
        "turnover": ["revenue", "sales", "gross income"],
        "authorized": ["approved", "sanctioned", "permitted", "certified"],
        "unauthorized": ["not approved", "unapproved", "prohibited", "illicit"],
        "approved": ["authorized", "sanctioned", "accepted"],
        "buyout": ["acquisition", "takeover", "merger"],
        "acquisition": ["buyout", "takeover"],
        "refused": ["rejected", "declined", "turned down"],
        "rejected": ["refused", "declined", "dismissed"],
    }

    @classmethod
    def _deterministic_grade(cls, question: str, documents: List[SearchResultChunk]) -> tuple[bool, str]:
        """
        Deterministic relevance grader checking keyword overlap, proposition-level negation,
        domain synonyms, and semantic entity presence to defend against adversarial traps.
        """
        q_lower = question.lower()
        q_words = set(re.findall(r"\b[a-zA-Z0-9_\-#]{3,}\b", q_lower))
        stop_words = {"what", "when", "where", "which", "how", "does", "specific", "parameters", "governing", "system", "tell", "show"}
        salient_words = q_words - stop_words
        if not salient_words:
            salient_words = q_words

        # Check for double negation in the query: e.g. "not unauthorized", "never failed", "not unapproved"
        double_neg_pattern = r"\b(not|never)\s+(un\w+|dis\w+|in\w+|im\w+|non\w+|failed|denied|refused)\b"
        has_double_negation = bool(re.search(double_neg_pattern, q_lower))

        # Check for single negation constraint in query
        negation_terms = {"not", "never", "refused", "denied", "failed", "unsuccessful", "non", "contraindicated"}
        q_has_single_negation = (not has_double_negation) and any(t in q_lower.split() for t in negation_terms)

        # Split documents into individual sentences to evaluate propositions locally
        all_sentences = []
        for d in documents:
            for s in re.split(r"(?<=[.!?])\s+", d.content):
                s_strip = s.strip()
                if s_strip:
                    all_sentences.append(s_strip)

        combined_doc_text = " ".join(all_sentences).lower()

        # Proposition-level negation defense:
        # If the query requires a negative condition (e.g. "Which country did not ratify..."),
        # we check the sentence(s) that talk about the target topic.
        if q_has_single_negation:
            topic_words = salient_words - negation_terms
            topic_sentences = []
            for s in all_sentences:
                s_lower = s.lower()
                if sum(1 for w in topic_words if w in s_lower) >= max(1, len(topic_words) // 2):
                    topic_sentences.append(s)

            if topic_sentences:
                topic_has_negation = any(
                    any(t in s.lower().split() for t in negation_terms)
                    for s in topic_sentences
                )
                if not topic_has_negation:
                    return False, "Query contains explicit negation constraint, but retrieved documents lack negation markers on the target topic (lexical trap detected)."
            else:
                return False, "Query contains explicit negation constraint, but retrieved documents lack negation markers (lexical trap detected)."

        # Match concept overlap with domain synonym expansion
        matched_concepts = set()
        for word in salient_words:
            if word in combined_doc_text:
                matched_concepts.add(word)
                continue
            synonyms = cls.DOMAIN_SYNONYMS.get(word, [])
            if any(syn in combined_doc_text for syn in synonyms):
                matched_concepts.add(word)

        # Check multi-word phrase synonyms
        for phrase, syn_list in cls.DOMAIN_SYNONYMS.items():
            if phrase in q_lower:
                for syn in syn_list:
                    if syn in combined_doc_text:
                        matched_concepts.add(phrase)
                        for pw in phrase.split():
                            if pw in salient_words:
                                matched_concepts.add(pw)

        ratio = len(matched_concepts) / max(1, len(salient_words))
        has_code_match = any(re.search(r"\b[A-Z]{2,}-\d+\b", question) and w in combined_doc_text for w in salient_words)

        if ratio >= 0.35 or has_code_match:
            return True, f"Documents contain {len(matched_concepts)}/{len(salient_words)} salient terms ({ratio:.1%})."
        return False, f"Retrieved documents lack core terms ({len(matched_concepts)}/{len(salient_words)} matched)."

    # =========================================================================
    # Node 4: Web Search Fallback
    # =========================================================================
    def web_search_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes external web search fallback when local documents fail grading.
        """
        query = state.get("rewritten_query") or state["question"]
        trace = list(state.get("trace", []))
        trace.append("web_search_fallback")

        logger.info(f"Initiating Web Search fallback for query: '{query}'")

        web_results = []
        if settings.WEB_SEARCH_ENABLED:
            try:
                from langchain_community.tools import DuckDuckGoSearchRun
                search_tool = DuckDuckGoSearchRun()
                raw_search = search_tool.run(query)
                if raw_search and len(raw_search.strip()) > 20:
                    web_results.append({
                        "title": f"Live Web Result for {query[:35]}",
                        "source": "duckduckgo.com",
                        "page": 1,
                        "type": "Web",
                        "content": raw_search[:800],
                    })
            except Exception as e:
                logger.warning(f"Live DuckDuckGo search unavailable or offline: {e}. Using deterministic fallback.")

        if not web_results:
            web_results.append({
                "title": f"[SIMULATED OFFLINE FALLBACK] Public Knowledge on {query[:30]}",
                "source": "simulated_offline_fallback",
                "page": 1,
                "type": "Web",
                "content": f"[SIMULATED OFFLINE FALLBACK] No live internet connection available. Synthetic mock reference for '{query}': specification parameters and public reference.",
            })

        logger.info(f"Web search retrieved {len(web_results)} external documents.")
        return {
            "web_search_needed": True,
            "web_documents": web_results,
            "trace": trace,
        }

    # =========================================================================
    # Node 5: Generator (Answer & Cite)
    # =========================================================================
    def generator_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Generates the final answer strictly enforcing the required citation syntax:
        [Source: FileName, Page X, Type: Table/Text]
        Applies ContextCompressor to prune irrelevant sentences before prompt injection.
        """
        question = state["question"]
        documents = state.get("documents", [])
        web_docs = state.get("web_documents", [])
        web_needed = state.get("web_search_needed", False)
        trace = list(state.get("trace", []))
        trace.append("generator")

        logger.info(f"Generating cited response for question: '{question}'")

        # Apply ContextCompressor on retrieved documents
        compressor = ContextCompressor()
        compressed_docs = compressor.compress_all(question, documents) if not web_needed else []

        # Format context for generator
        context_blocks = []
        if not web_needed and compressed_docs:
            for c_doc in compressed_docs:
                c_type = c_doc.modality_type.capitalize()
                tag = f"[Source: {c_doc.source_file}, Page {c_doc.page_number}, Type: {c_type}]"
                context_blocks.append(f"Passage ({tag}):\n{c_doc.compressed_content}")

        for w_doc in web_docs:
            tag = f"[Source: {w_doc['source']}, Page {w_doc['page']}, Type: {w_doc['type']}]"
            context_blocks.append(f"Web Passage ({tag}):\n{w_doc['content']}")

        full_context = "\n\n".join(context_blocks)

        if self._llm is not None:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"User Question: {question}\n\n"
                            f"Context Passages with Citations:\n{full_context}\n\n"
                            "Answer the question and strictly cite every claim using [Source: FileName, Page X, Type: Y]."
                        )
                    ),
                ]
                answer = self._llm.invoke(messages).content.strip()
            except Exception as e:
                logger.error(f"Error in LLM generator: {e}. Using deterministic cited synthesis.")
                answer = self._deterministic_generate(question, documents, web_docs, web_needed)
        else:
            answer = self._deterministic_generate(question, documents, web_docs, web_needed)

        # Parse and validate citations
        citations = self._extract_citations(answer, documents, web_docs)

        logger.info(f"Generated answer ({len(answer)} chars) with {len(citations)} verified citations.")
        return {
            "generation": answer,
            "citations": [c.model_dump() for c in citations],
            "trace": trace,
        }

    @staticmethod
    def _deterministic_generate(
        question: str,
        documents: List[SearchResultChunk],
        web_docs: List[Dict[str, Any]],
        web_search_needed: bool = False,
    ) -> str:
        """Synthesizes a response with guaranteed strict citations in offline/mock mode."""
        claims = []
        if web_search_needed and web_docs:
            w_doc = web_docs[0]
            w_source = w_doc.get("source", "simulated_offline_fallback")
            w_page = w_doc.get("page", 1)
            w_type = w_doc.get("type", "Web")
            w_citation = f"[Source: {w_source}, Page {w_page}, Type: {w_type}]"
            claims.append(f"Based on external documentation, {w_doc['content']} {w_citation}.")
        elif documents:
            top_doc = documents[0]
            c_type = top_doc.modality_type.capitalize()
            citation = f"[Source: {top_doc.source_file}, Page {top_doc.page_number}, Type: {c_type}]"
            first_sentence = top_doc.content.split(".")[0].strip()
            claims.append(f"According to verified documentation, {first_sentence} {citation}.")

            if len(documents) > 1:
                sec_doc = documents[1]
                sec_type = sec_doc.modality_type.capitalize()
                sec_citation = f"[Source: {sec_doc.source_file}, Page {sec_doc.page_number}, Type: {sec_type}]"
                sec_sentence = sec_doc.content.split(".")[0].strip()
                claims.append(f"Furthermore, {sec_sentence} {sec_citation}.")
        elif web_docs:
            w_doc = web_docs[0]
            w_source = w_doc.get("source", "simulated_offline_fallback")
            w_page = w_doc.get("page", 1)
            w_type = w_doc.get("type", "Web")
            w_citation = f"[Source: {w_source}, Page {w_page}, Type: {w_type}]"
            claims.append(f"Based on external documentation, {w_doc['content']} {w_citation}.")
        else:
            return "No relevant information could be retrieved to answer the question [Source: Unknown, Page 0, Type: Text]."

        return " ".join(claims)

    @staticmethod
    def _extract_citations(
        text: str,
        documents: List[SearchResultChunk],
        web_docs: List[Dict[str, Any]],
    ) -> List[CitationItem]:
        """
        Extracts and strictly verifies all citations adhering to [Source: File, Page X, Type: Y].
        Rejects hallucinated or fabricated citations that do not exist in retrieved evidence.
        """
        pattern = r"\[Source:\s*([^,]+),\s*Page\s*(\d+),\s*Type:\s*([^\]]+)\]"
        matches = re.finditer(pattern, text)

        # Build lookup tables for valid sources
        doc_map = {}
        for d in documents:
            norm_name = Path(d.source_file).name.lower()
            doc_map[(norm_name, d.page_number)] = d
            doc_map[(d.source_file.lower(), d.page_number)] = d

        web_sources = {
            w.get("source", "").lower(): w for w in web_docs
        }

        citations: List[CitationItem] = []
        seen_tags = set()

        for match in matches:
            source_file = match.group(1).strip()
            page_num = int(match.group(2).strip())
            c_type = match.group(3).strip()
            tag = f"[Source: {source_file}, Page {page_num}, Type: {c_type}]"

            if tag in seen_tags:
                continue

            # Grounding check: verify whether citation source exists in retrieved chunks or web results
            norm_source = Path(source_file).name.lower()
            key_name = (norm_source, page_num)
            key_full = (source_file.lower(), page_num)

            matched_chunk = doc_map.get(key_name) or doc_map.get(key_full)
            matched_web = web_sources.get(source_file.lower()) or web_sources.get(norm_source)

            if not matched_chunk and not matched_web:
                logger.warning(
                    f"Rejecting ungrounded/hallucinated citation '{tag}': "
                    f"source '{source_file}' (page {page_num}) was never retrieved."
                )
                continue

            bbox = None
            if matched_chunk and matched_chunk.bounding_box:
                bbox = matched_chunk.bounding_box.model_dump()

            seen_tags.add(tag)
            citations.append(
                CitationItem(
                    source_file=source_file,
                    page_number=page_num,
                    modality_type=c_type,
                    citation_tag=tag,
                    bounding_box=bbox,
                )
            )

        return citations
