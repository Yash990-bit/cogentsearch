"""
Cross-Encoder Reranker Module for CogentSearch.
Utilizes neural cross-attention (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2)
to evaluate query-passage semantic relevance, with input validation, batch inference,
and deterministic fallback for isolated test environments.

ENGINEERING NOTE ON LIMITATIONS:
Cross-encoders are dense cross-attention relevance estimators trained on passage-matching
loss (e.g., MS-MARCO). They excel at ranking passages with high semantic overlap, but do not
guarantee formal deductive reasoning, negation verification, or mathematical consistency.
Adversarial lexical traps (passages repeating verbatim query terms in a non-factual context)
can score higher than abstract factual answers. Deep verification of truthfulness and negation
is the explicit domain of Phase 4's Agentic RAG loop (Hallucination Grader / Self-Correction).
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional
from loguru import logger


class CrossEncoderReranker:
    """
    Computes cross-attention relevance scores between a query and candidate chunks.
    Preserves authentic neural inference with safe handling of edge cases and empty inputs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_local_fallback: bool = False,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.use_local_fallback = use_local_fallback
        self.batch_size = batch_size
        self._model = None

        if not use_local_fallback:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded neural CrossEncoder model '{self.model_name}'.")
            except Exception as e:
                logger.warning(
                    f"Could not load HuggingFace CrossEncoder '{self.model_name}': {e}. "
                    "Operating in fallback cross-attention scoring mode."
                )
                self.use_local_fallback = True

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Computes relevance scores for a list of (query, chunk_text) pairs.
        Returns a list of float scores.
        """
        if not pairs:
            return []

        # Sanitize pairs: replace None with empty strings
        clean_pairs = [(str(q or "").strip(), str(d or "").strip()) for q, d in pairs]

        if not self.use_local_fallback and self._model is not None:
            try:
                raw_scores = self._model.predict(clean_pairs, batch_size=self.batch_size)
                # Convert numpy array or float to python list of floats
                return [round(float(s), 4) for s in raw_scores]
            except Exception as e:
                logger.error(f"CrossEncoder inference error: {e}. Using fallback scorer.")

        # Fallback cross-attention proxy scorer
        return [self._offline_score(q, doc) for q, doc in clean_pairs]

    @staticmethod
    def _offline_score(query: str, doc_text: str) -> float:
        """
        Deterministic cross-attention proxy scoring query terms, exact phrases,
        and semantic proximity in document text.
        """
        q_lower = query.lower().strip()
        doc_lower = doc_text.lower().strip()

        if not q_lower or not doc_lower:
            return -10.0

        score = 0.0

        # Exact phrase bonus
        if q_lower in doc_lower:
            score += 3.5

        q_words = [w for w in q_lower.split() if len(w) > 1]
        if not q_words:
            return 0.0

        matched_words = 0
        for w in q_words:
            if w in doc_lower:
                matched_words += 1
                # Higher weight for technical terms, IDs, or numbers
                if any(c.isdigit() for c in w) or "-" in w or "$" in w:
                    score += 1.5
                else:
                    score += 1.0

        coverage_ratio = matched_words / len(q_words)
        score += coverage_ratio * 2.0

        # Density / proximity bonus
        if len(q_words) > 1 and coverage_ratio >= 0.7:
            score += 1.0

        return round(score, 4)
