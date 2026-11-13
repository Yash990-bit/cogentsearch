"""
Semantic Recursive Document Chunking Module for CogentSearch.
Implements structural boundary-aware recursive character splitting while preserving
atomic tabular blocks, visual element summaries, and page/coordinate metadata.
"""

from __future__ import annotations
import re
from typing import List, Optional, Dict, Any, Literal
from loguru import logger

from cogent.schemas.document import MultimodalDocument, ChunkType, BoundingBox
from cogent.schemas.indexing import IndexedDocument


class SemanticDocumentSplitter:
    """
    Splits MultimodalDocument instances into atomic, semantic IndexedDocument objects.
    - Preserves tables and visual blocks atomically.
    - Uses recursive character boundaries (Headings, Paragraphs, Sentences).
    - Injects modality_type ('text', 'table', 'image'), page_number, bounding_box, and source_file.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]

    def split_document(self, doc: MultimodalDocument) -> List[IndexedDocument]:
        """Converts a MultimodalDocument into a list of IndexedDocument chunks."""
        indexed_chunks: List[IndexedDocument] = []
        chunk_idx = 0

        # Create quick lookup for visual elements by id
        visual_lookup = {v.element_id: v for v in doc.visual_elements}

        for chunk in doc.chunks:
            # Map ChunkType to modality_type ('text', 'table', 'image')
            modality: Literal["text", "table", "image"] = "text"
            bbox: Optional[BoundingBox] = None

            if chunk.chunk_type == ChunkType.TABLE:
                modality = "table"
                if chunk.bounding_boxes:
                    bbox = chunk.bounding_boxes[0]
                elif chunk.visual_element_ids and chunk.visual_element_ids[0] in visual_lookup:
                    bbox = visual_lookup[chunk.visual_element_ids[0]].bounding_box

            elif chunk.chunk_type == ChunkType.VISUAL_SUMMARY:
                modality = "image"
                if chunk.bounding_boxes:
                    bbox = chunk.bounding_boxes[0]
                elif chunk.visual_element_ids and chunk.visual_element_ids[0] in visual_lookup:
                    bbox = visual_lookup[chunk.visual_element_ids[0]].bounding_box

            # Atomic Preservation: Tables and visual summaries should never be fragmented
            if modality in ["table", "image"] or len(chunk.content) <= self.chunk_size:
                indexed_chunks.append(
                    IndexedDocument(
                        chunk_id=f"{doc.document_id}_{chunk.chunk_id}_{chunk_idx}",
                        content=chunk.content.strip(),
                        modality_type=modality,
                        page_number=chunk.page_number,
                        bounding_box=bbox,
                        source_file=doc.filename,
                        headers_hierarchy=chunk.headers_hierarchy,
                        metadata={
                            **chunk.metadata,
                            "document_id": doc.document_id,
                            "original_chunk_id": chunk.chunk_id,
                        },
                    )
                )
                chunk_idx += 1
            else:
                # Textual content exceeding chunk_size is split recursively
                sub_chunks = self._recursive_split_text(chunk.content, self.separators)
                for sub_idx, sub_text in enumerate(sub_chunks):
                    if not sub_text.strip():
                        continue
                    indexed_chunks.append(
                        IndexedDocument(
                            chunk_id=f"{doc.document_id}_{chunk.chunk_id}_{chunk_idx}",
                            content=sub_text.strip(),
                            modality_type="text",
                            page_number=chunk.page_number,
                            bounding_box=None,
                            source_file=doc.filename,
                            headers_hierarchy=chunk.headers_hierarchy,
                            metadata={
                                **chunk.metadata,
                                "document_id": doc.document_id,
                                "original_chunk_id": chunk.chunk_id,
                                "sub_slice": sub_idx,
                            },
                        )
                    )
                    chunk_idx += 1

        logger.info(
            f"Split document '{doc.filename}' ({len(doc.chunks)} raw chunks) "
            f"into {len(indexed_chunks)} IndexedDocuments."
        )
        return indexed_chunks

    def _recursive_split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive character splitting algorithm."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._recursive_split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Hard character slicing fallback
                    start = 0
                    while start < len(s):
                        good_splits.append(s[start:start + self.chunk_size])
                        start += self.chunk_size - self.chunk_overlap

        # Merge small splits respecting chunk_size and overlap
        current_chunk = ""
        for s in good_splits:
            if not current_chunk:
                current_chunk = s
            elif len(current_chunk) + len(s) + len(separator) <= self.chunk_size:
                current_chunk += separator + s
            else:
                final_chunks.append(current_chunk)
                current_chunk = s

        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks

