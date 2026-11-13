"""
Vision Extractor Module for CogentSearch.
Handles bounding box localization, high-resolution rendering and cropping of tables and images,
and enrichment via the Vision LLM client with strict geometric validation and OCR fallback.
"""

from __future__ import annotations
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from loguru import logger
import fitz  # PyMuPDF

from cogent.core.config import settings
from cogent.schemas.document import BoundingBox, VisualElement, VisualElementType
from cogent.vision.llm_client import VisionLLMClient


class VisionExtractor:
    """Extracts, crops, and enriches visual assets (tables, images, charts) from documents."""

    def __init__(
        self,
        vision_client: Optional[VisionLLMClient] = None,
        dpi_scale: Optional[float] = None,
        min_width: Optional[int] = None,
        min_height: Optional[int] = None,
    ):
        self.vision_client = vision_client or VisionLLMClient()
        self.dpi_scale = dpi_scale or settings.DPI_SCALE
        self.min_width = min_width or settings.MIN_IMAGE_WIDTH
        self.min_height = min_height or settings.MIN_IMAGE_HEIGHT
        self.output_dir = settings.EXTRACTED_IMAGES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_pdf_page(
        self,
        doc_id: str,
        page: fitz.Page,
        page_number: int,
        page_text_context: str = "",
        previous_page_tables: Optional[List[VisualElement]] = None,
    ) -> List[VisualElement]:
        """
        Scans a PyMuPDF page for tables and images, crops them at high DPI,
        and annotates them using the Vision LLM with strict coordinate clipping.
        """
        visual_elements: List[VisualElement] = []
        page_rect = page.rect
        page_w = max(1.0, float(page_rect.width))
        page_h = max(1.0, float(page_rect.height))

        # 1. Detect and crop tables
        table_tuples = self._detect_tables(page, page_number, page_w, page_h)
        for idx, (bbox, raw_md_table, verbatim_text) in enumerate(table_tuples):
            elem_id = f"table_{doc_id}_p{page_number}_{idx}"
            crop_path = self._crop_bounding_box(page, bbox, elem_id)
            if not crop_path:
                continue

            # Query Vision LLM with table flag and context hint
            llm_result = self.vision_client.analyze_image(
                image_path=crop_path,
                context_hint=page_text_context[:500],
                is_table=True,
            )

            # Prioritize verbatim document table extraction; fallback to Vision LLM markdown reconstruction
            final_md = raw_md_table or llm_result.get("markdown_table")

            # Check if this table is a continuation of a table from the previous page
            continuation_id = None
            if previous_page_tables:
                for prev_t in previous_page_tables:
                    # If column widths or header matches previous page table
                    if prev_t.markdown_representation and final_md:
                        prev_header = prev_t.markdown_representation.splitlines()[0] if prev_t.markdown_representation.splitlines() else ""
                        curr_header = final_md.splitlines()[0] if final_md.splitlines() else ""
                        if prev_header and prev_header == curr_header:
                            continuation_id = prev_t.element_id
                            break

            visual_elements.append(
                VisualElement(
                    element_id=elem_id,
                    element_type=VisualElementType.TABLE,
                    page_number=page_number,
                    bounding_box=bbox,
                    image_path=str(crop_path),
                    description=llm_result["description"],
                    extracted_text=verbatim_text,
                    markdown_representation=final_md,
                    confidence=0.95,
                    continuation_of=continuation_id,
                    metadata={"keywords": llm_result.get("keywords", [])},
                )
            )

        # 2. Detect and crop images and charts
        existing_table_boxes = [b for b, _, _ in table_tuples]
        image_bboxes = self._detect_images(page, page_number, page_w, page_h, existing_table_boxes=existing_table_boxes)
        for idx, bbox in enumerate(image_bboxes):
            elem_id = f"img_{doc_id}_p{page_number}_{idx}"
            crop_path = self._crop_bounding_box(page, bbox, elem_id)
            if not crop_path:
                continue

            llm_result = self.vision_client.analyze_image(
                image_path=crop_path,
                context_hint=page_text_context[:500],
                is_table=False,
            )

            visual_elements.append(
                VisualElement(
                    element_id=elem_id,
                    element_type=llm_result.get("element_type", VisualElementType.IMAGE),
                    page_number=page_number,
                    bounding_box=bbox,
                    image_path=str(crop_path),
                    description=llm_result["description"],
                    extracted_text=None,
                    markdown_representation=llm_result.get("markdown_table"),
                    confidence=0.90,
                    metadata={"keywords": llm_result.get("keywords", [])},
                )
            )

        return visual_elements

    def _detect_tables(
        self, page: fitz.Page, page_number: int, page_w: float, page_h: float
    ) -> List[Tuple[BoundingBox, Optional[str], Optional[str]]]:
        """Identifies table bounding boxes, markdown representation, and verbatim cell text."""
        results: List[Tuple[BoundingBox, Optional[str], Optional[str]]] = []
        try:
            tabs = page.find_tables()
            for tab in tabs:
                bbox_raw = tab.bbox  # (x0, y0, x1, y1)
                x0 = max(0.0, float(bbox_raw[0]))
                y0 = max(0.0, float(bbox_raw[1]))
                x1 = min(page_w, float(bbox_raw[2]))
                y1 = min(page_h, float(bbox_raw[3]))

                width = x1 - x0
                height = y1 - y0
                if width < self.min_width or height < self.min_height or width <= 0 or height <= 0:
                    continue

                bbox = BoundingBox(
                    page_number=page_number,
                    x0=round(x0, 2),
                    y0=round(y0, 2),
                    x1=round(x1, 2),
                    y1=round(y1, 2),
                    page_width=page_w,
                    page_height=page_h,
                )

                # Extract markdown and verbatim text using high-precision cell-word localization
                raw_md = None
                verbatim_cells: List[str] = []
                try:
                    if hasattr(tab, "cells") and hasattr(tab, "row_count") and hasattr(tab, "col_count") and len(tab.cells) == tab.row_count * tab.col_count:
                        # Extract exact words inside each individual cell rectangle
                        raw_grid: List[List[str]] = []
                        for r in range(tab.row_count):
                            row_vals = []
                            for c in range(tab.col_count):
                                cell_idx = r * tab.col_count + c
                                cell_rect = tab.cells[cell_idx]
                                cell_words = [w[4] for w in page.get_text("words", clip=cell_rect)]
                                row_vals.append(" ".join(cell_words).strip())
                            raw_grid.append(row_vals)

                        # Filter out empty spacer columns
                        valid_cols = [c for c in range(tab.col_count) if any(raw_grid[r][c] for r in range(tab.row_count))]
                        if valid_cols and len(raw_grid) > 0:
                            cleaned_rows = [[raw_grid[r][c] for c in valid_cols] for r in range(tab.row_count)]
                            for row in cleaned_rows:
                                verbatim_cells.extend([c for c in row if c])
                            header = cleaned_rows[0]
                            md_lines = [
                                "| " + " | ".join(header) + " |",
                                "| " + " | ".join(["---"] * len(header)) + " |",
                            ]
                            for r in cleaned_rows[1:]:
                                md_lines.append("| " + " | ".join(r) + " |")
                            raw_md = "\n".join(md_lines)
                    else:
                        raw_rows = tab.extract()
                        if raw_rows and len(raw_rows) > 0:
                            cleaned_rows = [
                                [str(c or "").strip().replace("\n", " ") for c in row]
                                for row in raw_rows
                            ]
                            for row in cleaned_rows:
                                verbatim_cells.extend([c for c in row if c])

                            header = cleaned_rows[0]
                            divider = ["---"] * len(header)
                            md_lines = [
                                "| " + " | ".join(header) + " |",
                                "| " + " | ".join(divider) + " |",
                            ]
                            for r in cleaned_rows[1:]:
                                md_lines.append("| " + " | ".join(r) + " |")
                            raw_md = "\n".join(md_lines)
                except Exception as ex:
                    logger.debug(f"Direct table extraction failed on page {page_number}: {ex}")

                verbatim_str = " | ".join(verbatim_cells) if verbatim_cells else None
                results.append((bbox, raw_md, verbatim_str))
        except Exception as e:
            logger.debug(f"PyMuPDF table detection skipped or failed on page {page_number}: {e}")

        return results

    def _detect_images(
        self,
        page: fitz.Page,
        page_number: int,
        page_w: float,
        page_h: float,
        existing_table_boxes: List[BoundingBox],
    ) -> List[BoundingBox]:
        """Detects image bounding boxes clamped strictly inside page bounds."""
        image_boxes: List[BoundingBox] = []
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            for rect in rects:
                x0 = max(0.0, float(rect.x0))
                y0 = max(0.0, float(rect.y0))
                x1 = min(page_w, float(rect.x1))
                y1 = min(page_h, float(rect.y1))

                width = x1 - x0
                height = y1 - y0
                if width < self.min_width or height < self.min_height or width <= 0 or height <= 0:
                    continue

                bbox = BoundingBox(
                    page_number=page_number,
                    x0=round(x0, 2),
                    y0=round(y0, 2),
                    x1=round(x1, 2),
                    y1=round(y1, 2),
                    page_width=page_w,
                    page_height=page_h,
                )

                if self._overlaps_any(bbox, existing_table_boxes):
                    continue

                if not any(self._is_near_duplicate(bbox, existing) for existing in image_boxes):
                    image_boxes.append(bbox)

        return image_boxes

    def _crop_bounding_box(self, page: fitz.Page, bbox: BoundingBox, elem_id: str) -> Optional[Path]:
        """Crops the specified region at high DPI resolution and saves as PNG with strict validation."""
        try:
            page_w = page.rect.width
            page_h = page.rect.height
            padding = 2.0
            crop_rect = fitz.Rect(
                max(0.0, bbox.x0 - padding),
                max(0.0, bbox.y0 - padding),
                min(page_w, bbox.x1 + padding),
                min(page_h, bbox.y1 + padding),
            )

            if crop_rect.width <= 0 or crop_rect.height <= 0:
                return None

            mat = fitz.Matrix(self.dpi_scale, self.dpi_scale)
            pix = page.get_pixmap(matrix=mat, clip=crop_rect)

            out_path = self.output_dir / f"{elem_id}.png"
            pix.save(str(out_path))

            # Verify file exists and is not empty
            if out_path.exists() and out_path.stat().st_size > 0:
                return out_path
            return None
        except Exception as e:
            logger.error(f"Failed to crop bounding box {bbox} for {elem_id}: {e}")
            return None

    @staticmethod
    def _overlaps_any(box: BoundingBox, targets: List[BoundingBox], threshold: float = 0.5) -> bool:
        """Determines if a bounding box overlaps significantly with any target box."""
        for target in targets:
            dx = min(box.x1, target.x1) - max(box.x0, target.x0)
            dy = min(box.y1, target.y1) - max(box.y0, target.y0)
            if dx > 0 and dy > 0:
                intersection_area = dx * dy
                smaller_area = min(box.area, target.area)
                if smaller_area > 0 and (intersection_area / smaller_area) >= threshold:
                    return True
        return False

    @staticmethod
    def _is_near_duplicate(box1: BoundingBox, box2: BoundingBox, tol: float = 5.0) -> bool:
        """Checks if two bounding boxes are almost identical within a pixel tolerance."""
        return (
            abs(box1.x0 - box2.x0) < tol
            and abs(box1.y0 - box2.y0) < tol
            and abs(box1.x1 - box2.x1) < tol
            and abs(box1.y1 - box2.y1) < tol
        )
