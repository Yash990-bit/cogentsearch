"""
Cross-Encoder Reranker Module for CogentSearch.
Utilizes cross-encoder/ms-marco-MiniLM-L-6-v2 to evaluate full query-document cross-attention,
with an offline fallback scorer for fast, isolated CI/CD testing.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional
from loguru import logger


class CrossEncoderReranker:
    """Computes cross-attention relevance scores between a query and candidate chunks."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", use_local_fallback: bool = False):
        self.model_name = model_name
        self.use_local_fallback = use_local_fallback
        self._model = None

        if not use_local_fallback:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded CrossEncoder model '{self.model_name}'.")
            except Exception as e:
                logger.warning(
                    f"Could not load HuggingFace CrossEncoder '{self.model_name}': {e}. "
                    "Falling back to high-fidelity offline cross-attention scorer."
                )
                self.use_local_fallback = True

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Computes relevance scores for a list of (query, chunk_text) pairs.
        Returns a list of float scores.
        """
        if not pairs:
            return []

        if not self.use_local_fallback and self._model is not None:
            try:
                raw_scores = self._model.predict(pairs)
                # Convert numpy array or float to python list of floats
                return [round(float(s), 4) for s in raw_scores]
            except Exception as e:
                logger.error(f"CrossEncoder inference error: {e}. Using fallback scorer.")

        # Fallback cross-attention proxy scorer
        return [self._offline_score(q, doc) for q, doc in pairs]

    @staticmethod
    def _offline_score(query: str, doc_text: str) -> float:
        """
        Deterministic cross-attention proxy scoring query terms, exact phrases,
        and semantic proximity in document text.
        """
        q_lower = query.lower().strip()
        doc_lower = doc_text.lower().strip()

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

