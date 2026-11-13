"""
Context Compressor and Token Pruning Module for CogentSearch Phase 3.
Implements query-guided extractive compression and sentence pruning across
multimodal chunks (text, tables, images) while preserving strict line-of-sight
traceability to source_file, page_number, and bounding_box.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
from loguru import logger
from pydantic import BaseModel, Field

from cogent.schemas.indexing import SearchResultChunk
from cogent.schemas.document import BoundingBox


class CompressedChunk(BaseModel):
    """Represents a compressed, query-focused snippet with preserved provenance."""

    chunk_id: str
    original_content: str
    compressed_content: str
    compression_ratio: float
    modality_type: str
    page_number: int
    bounding_box: Optional[BoundingBox] = None
    source_file: str
    score: float = 0.0


class ContextCompressor:
    """
    Extractively prunes irrelevant sentences and non-matching table rows from
    retrieved chunks before injecting them into the Generator LLM prompt.
    """

    def __init__(
        self,
        max_sentences_per_chunk: int = 4,
        min_sentence_length: int = 15,
        target_compression_ratio: float = 0.65,
    ):
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.min_sentence_length = min_sentence_length
        self.target_compression_ratio = target_compression_ratio

    def compress_chunk(self, query: str, chunk: SearchResultChunk) -> CompressedChunk:
        """Compresses a single multimodal chunk based on relevance to query."""
        content = chunk.content.strip()
        modality = chunk.modality_type.lower()

        if modality == "table":
            compressed_text = self._compress_table(query, content)
        elif modality == "image":
            compressed_text = self._compress_image_description(query, content)
        else:
            compressed_text = self._compress_text(query, content)

        ratio = round(len(compressed_text) / max(1, len(content)), 4)
        score_val = getattr(chunk, "score", None) or chunk.cross_encoder_score or chunk.rrf_score or 0.0
        return CompressedChunk(
            chunk_id=chunk.chunk_id,
            original_content=content,
            compressed_content=compressed_text,
            compression_ratio=ratio,
            modality_type=chunk.modality_type,
            page_number=chunk.page_number,
            bounding_box=chunk.bounding_box,
            source_file=chunk.source_file,
            score=float(score_val),
        )

    def compress_all(self, query: str, chunks: List[SearchResultChunk]) -> List[CompressedChunk]:
        """Compresses a list of retrieved chunks, preserving provenance."""
        if not chunks:
            return []
        compressed = [self.compress_chunk(query, c) for c in chunks]
        logger.info(
            f"ContextCompressor pruned {len(chunks)} chunks: average compression ratio "
            f"{sum(c.compression_ratio for c in compressed)/len(compressed):.2%}"
        )
        return compressed

    DECISIVE_TERMS = {
        "fail", "failed", "failure", "fails",
        "pass", "passed", "passes",
        "abort", "aborted", "aborts",
        "verdict", "approved", "reject", "rejected",
        "safe", "unsafe", "denied", "halt", "halted",
        "yes", "no", "cancelled", "canceled", "prohibited", "banned",
        "contraindicated", "refused", "terminated", "critical"
    }

    NEGATION_TERMS = {
        "not", "no", "never", "neither", "nor", "none",
        "without", "excluding", "except", "cannot", "non",
        "does not", "did not", "will not", "is not", "was not"
    }

    def _compress_text(self, query: str, text: str) -> str:
        """
        Extracts top scoring sentences matching query terms while strictly
        preserving decisive outcomes, safety verdicts, qualifications, and negations.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 2:
            return text

        q_terms = set(re.findall(r"\b[a-zA-Z0-9_\-#]{3,}\b", query.lower()))
        scored_sentences = []

        for idx, s in enumerate(sentences):
            s_clean = s.strip()
            if not s_clean:
                continue

            s_words = set(re.findall(r"\b[a-zA-Z0-9_\-#]{2,}\b", s_clean.lower()))
            s_lower = s_clean.lower()

            # Check for decisive statements, outcomes, or critical verdicts
            has_decisive = bool(s_words & self.DECISIVE_TERMS) or any(
                re.search(rf"\b{term}\b", s_lower) for term in ["verdict: no", "verdict: yes", "not safe"]
            )
            has_negation = bool(s_words & {"not", "never", "no", "none", "without", "except"}) or any(
                phrase in s_lower for phrase in ["does not", "did not", "is not", "was not", "will not"]
            )
            has_numbers = bool(re.search(r"(\$|€|£)?\b\d+([.,]\d+)?\b%?", s_clean))

            # Only drop sentences below minimum length if they lack decisive content, negations, or numbers
            if len(s_clean) < self.min_sentence_length:
                if not (has_decisive or has_negation or has_numbers):
                    continue

            overlap = len(q_terms & s_words)
            # Give slight preference to opening sentence for context
            pos_bonus = 0.5 if idx == 0 else 0.0

            # Decisive and negation bonuses ensure critical polarity is never silently dropped
            decisive_bonus = 2.5 if has_decisive else 0.0
            negation_bonus = 1.5 if (has_negation and (overlap > 0 or has_numbers)) else 0.0

            score = overlap + pos_bonus + decisive_bonus + negation_bonus
            scored_sentences.append((score, idx, s_clean))

        if not scored_sentences:
            return text

        # Sort by score descending, take top N
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_sentences[: self.max_sentences_per_chunk]

        # Re-sort chronologically by original sentence index
        top_k.sort(key=lambda x: x[1])
        return " ".join(s[2] for s in top_k)

    @staticmethod
    def _compress_table(query: str, markdown_table: str) -> str:
        """Preserves table header row and filters data rows matching query keywords."""
        lines = [l.strip() for l in markdown_table.split("\n") if l.strip()]
        if len(lines) <= 4:
            return markdown_table

        # Header and divider lines
        header = lines[0]
        divider = lines[1] if len(lines) > 1 and "---" in lines[1] else "| --- |"
        data_rows = lines[2:] if divider in lines else lines[1:]

        q_terms = set(re.findall(r"\b[a-zA-Z0-9_\-#]{3,}\b", query.lower()))
        scored_rows = []

        for r in data_rows:
            r_lower = r.lower()
            matches = sum(1 for t in q_terms if t in r_lower)
            scored_rows.append((matches, r))

        # Filter rows with matches; if none match, keep first 3 rows
        matching_rows = [r for score, r in scored_rows if score > 0]
        if not matching_rows:
            matching_rows = data_rows[:3]

        return "\n".join([header, divider] + matching_rows)

    @staticmethod
    def _compress_image_description(query: str, description: str) -> str:
        """Extracts key bullet points and summaries from visual element descriptions."""
        lines = [l.strip() for l in description.split("\n") if l.strip()]
        if len(lines) <= 3:
            return description

        # Retain summary and key data point lines
        salient_lines = []
        for l in lines:
            if any(k in l.lower() for k in ["summary", "key_data", "metrics", "chart", "table", "element_type"]):
                salient_lines.append(l)
            elif l.startswith("- ") or l.startswith("* "):
                salient_lines.append(l)

        if salient_lines:
            return "\n".join(salient_lines[:5])
        return description
