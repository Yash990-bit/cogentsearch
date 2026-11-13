"""
Embeddings Engine Module for CogentSearch.
Provides unified dense embedding interfaces supporting OpenAI (text-embedding-3-small)
and offline deterministic fallback with L2 normalization.
"""

from __future__ import annotations
import math
import hashlib
from typing import List, Optional
from loguru import logger

from cogent.core.config import settings
from cogent.core.exceptions import EmbeddingAPIError


class EmbeddingEngine:
    """Computes dense vector representations for document chunks and user queries."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: int = 1536,
        api_key: Optional[str] = None,
        allow_fallback: bool = False,
    ):
        self.provider = provider or settings.EMBEDDING_PROVIDER
        self.model_name = model_name or settings.OPENAI_EMBEDDING_MODEL
        self.dimension = dimension
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.allow_fallback = allow_fallback

        if self.provider == "openai" and not self.api_key:
            logger.warning(
                "No OPENAI_API_KEY detected for EmbeddingEngine. "
                "Operating in offline deterministic vector mode for local testing and CI."
            )
            self.provider = "mock"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Computes embeddings for a batch of text chunks."""
        if not texts:
            return []

        if self.provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                response = client.embeddings.create(
                    input=texts,
                    model=self.model_name,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if self.allow_fallback:
                    logger.warning(f"OpenAI embedding call failed: {e}. Falling back to offline embeddings (allow_fallback=True).")
                    return [self._offline_embed(t) for t in texts]
                logger.error(f"OpenAI embedding call failed: {e}. Raising EmbeddingAPIError to prevent vector space contamination.")
                raise EmbeddingAPIError(
                    f"OpenAI embedding inference failed: {e}. "
                    "Aborting operation to prevent mixing incompatible hash vectors into neural vector space."
                ) from e
        else:
            return [self._offline_embed(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Computes embedding for a single query string."""
        return self.embed_documents([query])[0]

    def _offline_embed(self, text: str) -> List[float]:
        """
        Deterministic, offline vectorizer generating normalized unit embeddings.
        Combines token bag-of-words hashing with character n-gram projections.
        Guarantees:
        1. Cosine similarity between identical texts is 1.0.
        2. Texts sharing keywords have high positive dot products.
        """
        words = text.lower().split()
        vector = [0.0] * self.dimension

        if not words:
            # Unit vector along first dimension for empty text
            vector[0] = 1.0
            return vector

        for word in words:
            # Deterministic bucket index via sha256
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vector[idx] += sign * 1.5

            # Secondary character trigram buckets
            for i in range(max(0, len(word) - 2)):
                tri = word[i:i+3]
                h_tri = int(hashlib.md5(tri.encode("utf-8")).hexdigest(), 16)
                idx_tri = h_tri % self.dimension
                vector[idx_tri] += 0.5

        # L2 Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            return [round(v / norm, 6) for v in vector]
        return vector

