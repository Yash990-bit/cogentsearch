"""
Vision Extractor Module for CogentSearch.
Handles bounding box localization, high-resolution rendering and cropping of tables and images,
and enrichment via the Vision LLM client.
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
    ) -> List[VisualElement]:
        """
        Scans a PyMuPDF page for tables and images, crops them at high DPI,
        and annotates them using the Vision LLM.
        """
        visual_elements: List[VisualElement] = []
        page_rect = page.rect
        page_w = page_rect.width
        page_h = page_rect.height

        # 1. Detect and crop tables
        table_bboxes = self._detect_tables(page, page_number, page_w, page_h)
        for idx, (bbox, raw_md_table) in enumerate(table_bboxes):
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

            # Use LLM markdown if provided, or fallback to detected table markdown
            final_md = llm_result.get("markdown_table") or raw_md_table

            visual_elements.append(
                VisualElement(
                    element_id=elem_id,
                    element_type=VisualElementType.TABLE,
                    page_number=page_number,
                    bounding_box=bbox,
                    image_path=str(crop_path),
                    description=llm_result["description"],
                    markdown_representation=final_md,
                    confidence=0.95,
                    metadata={"keywords": llm_result.get("keywords", [])},
                )
            )

        # 2. Detect and crop images and charts
        image_bboxes = self._detect_images(page, page_number, page_w, page_h, existing_table_boxes=[b for b, _ in table_bboxes])
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
                    markdown_representation=llm_result.get("markdown_table"),
                    confidence=0.90,
                    metadata={"keywords": llm_result.get("keywords", [])},
                )
            )

        return visual_elements

    def _detect_tables(
        self, page: fitz.Page, page_number: int, page_w: float, page_h: float
    ) -> List[Tuple[BoundingBox, Optional[str]]]:
        """Identifies table bounding boxes and preliminary markdown using PyMuPDF table finder."""
        results: List[Tuple[BoundingBox, Optional[str]]] = []
        try:
            tabs = page.find_tables()
            for tab in tabs:
                bbox_raw = tab.bbox  # (x0, y0, x1, y1)
                width = bbox_raw[2] - bbox_raw[0]
                height = bbox_raw[3] - bbox_raw[1]
                if width < self.min_width or height < self.min_height:
                    continue

                bbox = BoundingBox(
                    page_number=page_number,
                    x0=float(bbox_raw[0]),
                    y0=float(bbox_raw[1]),
                    x1=float(bbox_raw[2]),
                    y1=float(bbox_raw[3]),
                    page_width=page_w,
                    page_height=page_h,
                )

                # Extract markdown directly from table structure
                raw_md = None
                try:
                    # Native PyMuPDF table row extraction
                    raw_rows = tab.extract()
                    if raw_rows and len(raw_rows) > 0:
                        cleaned_rows = [
                            [str(c or "").strip().replace("\n", " ") for c in row]
                            for row in raw_rows
                        ]
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
                    logger.debug(f"Direct table extraction failed: {ex}")

                results.append((bbox, raw_md))
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
        """Detects image bounding boxes while filtering out duplicates or areas overlapping tables."""
        image_boxes: List[BoundingBox] = []
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            for rect in rects:
                width = rect.width
                height = rect.height
                if width < self.min_width or height < self.min_height:
                    continue

                # Check overlap with detected tables
                bbox = BoundingBox(
                    page_number=page_number,
                    x0=float(rect.x0),
                    y0=float(rect.y0),
                    x1=float(rect.x1),
                    y1=float(rect.y1),
                    page_width=page_w,
                    page_height=page_h,
                )

                if self._overlaps_any(bbox, existing_table_boxes):
                    continue

                # Avoid duplicate identical bounding boxes
                if not any(self._is_near_duplicate(bbox, existing) for existing in image_boxes):
                    image_boxes.append(bbox)

        return image_boxes

    def _crop_bounding_box(self, page: fitz.Page, bbox: BoundingBox, elem_id: str) -> Optional[Path]:
        """Crops the specified region at high DPI resolution and saves as PNG."""
        try:
            # Add small padding around the bounding box
            padding = 4.0
            crop_rect = fitz.Rect(
                max(0, bbox.x0 - padding),
                max(0, bbox.y0 - padding),
                min(page.rect.width, bbox.x1 + padding),
                min(page.rect.height, bbox.y1 + padding),
            )

            # High resolution matrix scaling
            mat = fitz.Matrix(self.dpi_scale, self.dpi_scale)
            pix = page.get_pixmap(matrix=mat, clip=crop_rect)

            out_path = self.output_dir / f"{elem_id}.png"
            pix.save(str(out_path))
            return out_path
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

