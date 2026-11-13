"""
Multimodal Ingestion Pipeline for CogentSearch.
Coordinates document parsing, multimodal visual reasoning, heading hierarchy tracking,
and schema-level visual injection to create fully enriched MultimodalDocument instances.
Features retry logic, strict provenance linking, and thread-safe execution.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

from cogent.core.config import settings
from cogent.core.exceptions import DocumentIngestionError
from cogent.schemas.document import (
    MultimodalDocument,
    DocumentChunk,
    ChunkType,
    VisualElement,
    VisualElementType,
)
from cogent.ingestion.parser import UniversalParserFactory, RawParsedDocument


class MultimodalIngestionPipeline:
    """End-to-end ingestion orchestrator for PDF, Word, and PowerPoint files."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        inject_visual_summaries: bool = True,
        max_retries: int = 2,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.inject_visual_summaries = inject_visual_summaries
        self.max_retries = max_retries
        self.processed_dir = settings.PROCESSED_DOCS_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_file(self, file_path: Path | str) -> MultimodalDocument:
        """
        Executes the complete Phase 1 ingestion workflow on a single document with retry support:
        1. Selects parser based on format (PDF, DOCX, PPTX).
        2. Extracts raw text flows, heading hierarchies, and page numbers.
        3. Identifies, crops, and enriches visual assets (tables, images, charts).
        4. Fuses visual descriptions & bounding boxes into semantic document chunks.
        5. Persists the MultimodalDocument contract to disk.
        """
        path = Path(file_path).resolve()
        logger.info(f"Initiating ingestion pipeline for: {path.name}")

        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._execute_ingestion(path, attempt)
            except DocumentIngestionError as die:
                # Do not retry on fundamental document validation errors (e.g. empty, password protected, unsupported)
                logger.error(f"Validation failure for '{path.name}': {die}")
                raise
            except Exception as e:
                last_exception = e
                logger.warning(f"Ingestion attempt {attempt + 1} failed for '{path.name}': {e}")
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue

        logger.error(f"Exhausted all retries ({self.max_retries}) for '{path.name}'.")
        raise last_exception or RuntimeError(f"Failed to process '{path.name}'")

    def _execute_ingestion(self, path: Path, attempt_num: int) -> MultimodalDocument:
        """Internal execution worker."""
        # 1. Parse document
        parser = UniversalParserFactory.get_parser(path)
        raw_doc: RawParsedDocument = parser.parse(path)

        # 2. Group visual elements by page for localized injection
        visuals_by_page: Dict[int, List[VisualElement]] = {}
        for elem in raw_doc.visual_elements:
            visuals_by_page.setdefault(elem.page_number, []).append(elem)

        # 3. Create semantic chunks with heading hierarchy & vision fusion
        chunks: List[DocumentChunk] = []
        chunk_counter = 0

        for page_num in range(1, raw_doc.total_pages + 1):
            page_text = raw_doc.page_texts.get(page_num, "").strip()
            headers = raw_doc.page_headers.get(page_num, [])
            page_visuals = visuals_by_page.get(page_num, [])

            # 3a. Generate chunks for textual content on the page
            if page_text:
                page_text_slices = self._sliding_window_chunks(page_text, self.chunk_size, self.chunk_overlap)
                for slice_idx, text_slice in enumerate(page_text_slices):
                    chunk_id = f"chunk_{raw_doc.document_id}_p{page_num}_{chunk_counter}"
                    chunk_counter += 1

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            page_number=page_num,
                            content=text_slice,
                            raw_text=text_slice,
                            injected_visual_descriptions=[],
                            chunk_type=ChunkType.TEXT,
                            headers_hierarchy=headers,
                            bounding_boxes=[],
                            visual_element_ids=[],
                            metadata={
                                "source": raw_doc.filename,
                                "slice_index": slice_idx,
                            },
                        )
                    )

            # 3b. Inject visual element summaries as dedicated retrieval chunks
            if self.inject_visual_summaries and page_visuals:
                for elem in page_visuals:
                    chunk_id = f"chunk_{raw_doc.document_id}_p{page_num}_{chunk_counter}"
                    chunk_counter += 1

                    is_table = elem.element_type == VisualElementType.TABLE
                    chunk_type = ChunkType.TABLE if is_table else ChunkType.VISUAL_SUMMARY

                    # Format enriched multimodal representation
                    enriched_content_parts = [
                        f"### [Visual Asset on Page {page_num}: {elem.element_type.value.upper()}]",
                        f"**Bounding Box Coordinates:** (x0={elem.bounding_box.x0:.1f}, y0={elem.bounding_box.y0:.1f}, x1={elem.bounding_box.x1:.1f}, y1={elem.bounding_box.y1:.1f})",
                        f"**Semantic Description:** {elem.description}",
                    ]

                    if elem.markdown_representation:
                        enriched_content_parts.append(f"\n**Tabular Data:**\n{elem.markdown_representation}")

                    enriched_content = "\n\n".join(enriched_content_parts)

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            page_number=page_num,
                            content=enriched_content,
                            raw_text=elem.extracted_text,
                            injected_visual_descriptions=[elem.description] if elem.description else [],
                            chunk_type=chunk_type,
                            headers_hierarchy=headers + [f"Visual: {elem.element_type.value}"],
                            bounding_boxes=[elem.bounding_box],
                            visual_element_ids=[elem.element_id],
                            metadata={
                                "source": raw_doc.filename,
                                "element_id": elem.element_id,
                                "element_type": elem.element_type.value,
                                "image_path": elem.image_path,
                            },
                        )
                    )

        # 4. Construct final MultimodalDocument
        multimodal_doc = MultimodalDocument(
            document_id=raw_doc.document_id,
            filename=raw_doc.filename,
            file_path=raw_doc.file_path,
            file_type=raw_doc.file_type,
            total_pages=raw_doc.total_pages,
            chunks=chunks,
            visual_elements=raw_doc.visual_elements,
            raw_metadata={
                **raw_doc.raw_metadata,
                "ingestion_retries": attempt_num,
            },
        )

        # 5. Persist serialized document JSON
        output_file = self.processed_dir / f"{raw_doc.document_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(multimodal_doc.model_dump_json(indent=2))

        logger.info(
            f"Successfully processed '{raw_doc.filename}'. "
            f"Pages: {raw_doc.total_pages} | Chunks: {len(chunks)} | Visuals: {len(raw_doc.visual_elements)}. "
            f"Saved to: {output_file.name}"
        )

        return multimodal_doc

    @staticmethod
    def _sliding_window_chunks(text: str, max_chars: int, overlap: int) -> List[str]:
        """Splits page text into character-bounded sliding windows respecting paragraph breaks."""
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        final_chunks: List[str] = []
        for c in chunks:
            if len(c) > max_chars * 1.5:
                start = 0
                while start < len(c):
                    end = start + max_chars
                    final_chunks.append(c[start:end])
                    start = end - overlap
            else:
                final_chunks.append(c)

        return final_chunks
