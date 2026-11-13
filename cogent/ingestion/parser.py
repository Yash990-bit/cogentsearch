"""
Universal Document Parser Module for CogentSearch.
Provides specialized parsers for PDF, DOCX (Word), and PPTX (PowerPoint) files,
preserving layout hierarchy, heading trees, page/slide numbers, and multimodal assets.
"""

from __future__ import annotations
import io
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import fitz  # PyMuPDF
from PIL import Image

from cogent.core.config import settings
from cogent.schemas.document import (
    BoundingBox,
    VisualElement,
    VisualElementType,
    DocumentChunk,
    ChunkType,
)
from cogent.vision.extractor import VisionExtractor
from cogent.vision.llm_client import VisionLLMClient


class RawParsedDocument:
    """Intermediate representation produced by document parsers before final chunking and fusion."""

    def __init__(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        file_type: str,
        total_pages: int,
        page_texts: Dict[int, str],
        page_headers: Dict[int, List[str]],
        visual_elements: List[VisualElement],
        raw_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.document_id = document_id
        self.filename = filename
        self.file_path = file_path
        self.file_type = file_type
        self.total_pages = total_pages
        self.page_texts = page_texts
        self.page_headers = page_headers
        self.visual_elements = visual_elements
        self.raw_metadata = raw_metadata or {}


class BaseDocumentParser(ABC):
    """Abstract base class for format-specific parsers."""

    @abstractmethod
    def parse(self, file_path: Path | str) -> RawParsedDocument:
        """Parse a document into structured page blocks and visual elements."""
        pass


class PDFParser(BaseDocumentParser):
    """High-fidelity PDF parser using PyMuPDF and layout-aware heading detection."""

    def __init__(self, vision_extractor: Optional[VisionExtractor] = None):
        self.vision_extractor = vision_extractor or VisionExtractor()

    def parse(self, file_path: Path | str) -> RawParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        file_bytes = path.read_bytes()
        doc_id = hashlib.sha256(file_bytes[:4096] + str(path.stat().st_size).encode()).hexdigest()[:12]

        doc = fitz.open(str(path))
        total_pages = len(doc)
        page_texts: Dict[int, str] = {}
        page_headers: Dict[int, List[str]] = {}
        all_visual_elements: List[VisualElement] = []

        logger.info(f"Parsing PDF '{path.name}' ({total_pages} pages) with layout & vision extraction...")

        # Determine average body text font size to accurately detect headings
        body_font_size = self._estimate_body_font_size(doc)

        current_hierarchy: List[str] = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Structural text and header extraction
            page_text_lines: List[str] = []
            page_detected_headers: List[str] = []

            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        line_text = "".join(s.get("text", "") for s in spans).strip()
                        if not line_text:
                            continue

                        # Check if line qualifies as a header (larger font or bold)
                        max_span_size = max((s.get("size", 10.0) for s in spans), default=10.0)
                        is_bold = any("bold" in s.get("font", "").lower() for s in spans)

                        if max_span_size >= body_font_size * 1.25 or (is_bold and len(line_text) < 80):
                            header_tag = f"### {line_text}" if max_span_size < body_font_size * 1.5 else f"## {line_text}"
                            page_text_lines.append(f"\n{header_tag}\n")
                            page_detected_headers.append(line_text)
                            current_hierarchy = [line_text]
                        else:
                            page_text_lines.append(line_text)

            combined_page_text = "\n".join(page_text_lines).strip()
            page_texts[page_num] = combined_page_text
            page_headers[page_num] = list(current_hierarchy)

            # 2. Extract visual elements (tables, charts, images)
            page_visuals = self.vision_extractor.extract_from_pdf_page(
                doc_id=doc_id,
                page=page,
                page_number=page_num,
                page_text_context=combined_page_text,
            )
            all_visual_elements.extend(page_visuals)

        metadata = doc.metadata or {}
        doc.close()

        return RawParsedDocument(
            document_id=doc_id,
            filename=path.name,
            file_path=str(path.resolve()),
            file_type="pdf",
            total_pages=total_pages,
            page_texts=page_texts,
            page_headers=page_headers,
            visual_elements=all_visual_elements,
            raw_metadata=metadata,
        )

    @staticmethod
    def _estimate_body_font_size(doc: fitz.Document) -> float:
        """Computes modal font size across the document sample to calibrate header thresholds."""
        sizes: List[float] = []
        for i in range(min(5, len(doc))):
            page_dict = doc[i].get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            sizes.append(round(span.get("size", 11.0), 1))
        if not sizes:
            return 11.0
        # Return most frequent font size
        return max(set(sizes), key=sizes.count)


class DocxParser(BaseDocumentParser):
    """Parser for Microsoft Word (.docx) documents."""

    def __init__(self, vision_client: Optional[VisionLLMClient] = None):
        self.vision_client = vision_client or VisionLLMClient()
        self.output_dir = settings.EXTRACTED_IMAGES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, file_path: Path | str) -> RawParsedDocument:
        import docx

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {path}")

        file_bytes = path.read_bytes()
        doc_id = hashlib.sha256(file_bytes[:4096] + str(path.stat().st_size).encode()).hexdigest()[:12]

        doc = docx.Document(str(path))
        logger.info(f"Parsing DOCX '{path.name}'...")

        page_texts: Dict[int, str] = {1: ""}
        page_headers: Dict[int, List[str]] = {1: []}
        visual_elements: List[VisualElement] = []

        text_lines: List[str] = []
        current_headers: List[str] = []

        # 1. Extract paragraphs and headers
        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue

            style_name = p.style.name.lower() if p.style else ""
            if "heading 1" in style_name:
                text_lines.append(f"\n# {text}\n")
                current_headers = [text]
            elif "heading 2" in style_name:
                text_lines.append(f"\n## {text}\n")
                current_headers = [current_headers[0] if current_headers else "", text]
            elif "heading" in style_name:
                text_lines.append(f"\n### {text}\n")
                current_headers.append(text)
            else:
                text_lines.append(text)

        # 2. Extract embedded tables
        for idx, table in enumerate(doc.tables):
            elem_id = f"table_{doc_id}_p1_{idx}"
            table_rows = []
            for row in table.rows:
                table_rows.append([cell.text.strip().replace("\n", " ") for cell in row.cells])

            if table_rows:
                header = table_rows[0]
                divider = ["---"] * len(header)
                md_table_lines = [
                    "| " + " | ".join(header) + " |",
                    "| " + " | ".join(divider) + " |",
                ]
                for r in table_rows[1:]:
                    md_table_lines.append("| " + " | ".join(r) + " |")

                md_table = "\n".join(md_table_lines)
                text_lines.append(f"\n{md_table}\n")

                visual_elements.append(
                    VisualElement(
                        element_id=elem_id,
                        element_type=VisualElementType.TABLE,
                        page_number=1,
                        bounding_box=BoundingBox(page_number=1, x0=0, y0=0, x1=100, y1=100),
                        image_path="",
                        description=f"Extracted Word Document Table: {elem_id}",
                        markdown_representation=md_table,
                        confidence=1.0,
                    )
                )

        # 3. Extract embedded media/images
        img_idx = 0
        for rel_id, rel in doc.part.related_parts.items():
            if "image" in rel.content_type:
                img_data = rel.blob
                elem_id = f"img_{doc_id}_p1_{img_idx}"
                img_path = self.output_dir / f"{elem_id}.png"
                img_path.write_bytes(img_data)

                llm_res = self.vision_client.analyze_image(
                    image_path=img_path,
                    context_hint=text_lines[0] if text_lines else "",
                    is_table=False,
                )

                visual_elements.append(
                    VisualElement(
                        element_id=elem_id,
                        element_type=llm_res.get("element_type", VisualElementType.IMAGE),
                        page_number=1,
                        bounding_box=BoundingBox(page_number=1, x0=0, y0=0, x1=100, y1=100),
                        image_path=str(img_path),
                        description=llm_res["description"],
                        confidence=0.90,
                    )
                )
                img_idx += 1

        page_texts[1] = "\n".join(text_lines)
        page_headers[1] = current_headers

        return RawParsedDocument(
            document_id=doc_id,
            filename=path.name,
            file_path=str(path.resolve()),
            file_type="docx",
            total_pages=1,
            page_texts=page_texts,
            page_headers=page_headers,
            visual_elements=visual_elements,
            raw_metadata={"core_properties": str(doc.core_properties)},
        )


class PptxParser(BaseDocumentParser):
    """Parser for PowerPoint (.pptx) presentations preserving slide hierarchy and visual assets."""

    def __init__(self, vision_client: Optional[VisionLLMClient] = None):
        self.vision_client = vision_client or VisionLLMClient()
        self.output_dir = settings.EXTRACTED_IMAGES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, file_path: Path | str) -> RawParsedDocument:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PPTX file not found: {path}")

        file_bytes = path.read_bytes()
        doc_id = hashlib.sha256(file_bytes[:4096] + str(path.stat().st_size).encode()).hexdigest()[:12]

        prs = Presentation(str(path))
        total_slides = len(prs.slides)
        logger.info(f"Parsing PPTX '{path.name}' ({total_slides} slides)...")

        page_texts: Dict[int, str] = {}
        page_headers: Dict[int, List[str]] = {}
        visual_elements: List[VisualElement] = []

        slide_width = prs.slide_width.pt if prs.slide_width else 720.0
        slide_height = prs.slide_height.pt if prs.slide_height else 540.0

        for slide_idx, slide in enumerate(prs.slides):
            slide_num = slide_idx + 1
            slide_lines: List[str] = []
            slide_title = ""

            # Check for slide title
            if slide.shapes.title and slide.shapes.title.text:
                slide_title = slide.shapes.title.text.strip()
                slide_lines.append(f"# Slide {slide_num}: {slide_title}\n")

            img_idx = 0
            for shape in slide.shapes:
                # Shape coordinates
                x0 = shape.left.pt if shape.left else 0.0
                y0 = shape.top.pt if shape.top else 0.0
                x1 = x0 + (shape.width.pt if shape.width else 0.0)
                y1 = y0 + (shape.height.pt if shape.height else 0.0)
                bbox = BoundingBox(
                    page_number=slide_num,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    page_width=slide_width,
                    page_height=slide_height,
                )

                # Extract Text Frame
                if shape.has_text_frame and shape != slide.shapes.title:
                    for p in shape.text_frame.paragraphs:
                        p_text = p.text.strip()
                        if p_text:
                            slide_lines.append(f"- {p_text}")

                # Extract Table
                elif shape.has_table:
                    elem_id = f"table_{doc_id}_s{slide_num}_{img_idx}"
                    table = shape.table
                    rows_data = []
                    for row in table.rows:
                        rows_data.append([cell.text.strip().replace("\n", " ") for cell in row.cells])

                    if rows_data:
                        header = rows_data[0]
                        divider = ["---"] * len(header)
                        md_table = (
                            "| " + " | ".join(header) + " |\n"
                            + "| " + " | ".join(divider) + " |\n"
                            + "\n".join("| " + " | ".join(r) + " |" for r in rows_data[1:])
                        )
                        slide_lines.append(f"\n{md_table}\n")

                        visual_elements.append(
                            VisualElement(
                                element_id=elem_id,
                                element_type=VisualElementType.TABLE,
                                page_number=slide_num,
                                bounding_box=bbox,
                                image_path="",
                                description=f"Presentation Table on Slide {slide_num}: {slide_title}",
                                markdown_representation=md_table,
                                confidence=1.0,
                            )
                        )
                    img_idx += 1

                # Extract Picture / Image
                elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    elem_id = f"img_{doc_id}_s{slide_num}_{img_idx}"
                    image = shape.image
                    img_bytes = image.blob
                    img_path = self.output_dir / f"{elem_id}.{image.ext}"
                    img_path.write_bytes(img_bytes)

                    llm_res = self.vision_client.analyze_image(
                        image_path=img_path,
                        context_hint=f"Slide {slide_num}: {slide_title}",
                        is_table=False,
                    )

                    visual_elements.append(
                        VisualElement(
                            element_id=elem_id,
                            element_type=llm_res.get("element_type", VisualElementType.IMAGE),
                            page_number=slide_num,
                            bounding_box=bbox,
                            image_path=str(img_path),
                            description=llm_res["description"],
                            markdown_representation=llm_res.get("markdown_table"),
                            confidence=0.90,
                            metadata={"keywords": llm_res.get("keywords", [])},
                        )
                    )
                    img_idx += 1

            page_texts[slide_num] = "\n".join(slide_lines).strip()
            page_headers[slide_num] = [slide_title] if slide_title else [f"Slide {slide_num}"]

        return RawParsedDocument(
            document_id=doc_id,
            filename=path.name,
            file_path=str(path.resolve()),
            file_type="pptx",
            total_pages=total_slides,
            page_texts=page_texts,
            page_headers=page_headers,
            visual_elements=visual_elements,
            raw_metadata={},
        )


class UniversalParserFactory:
    """Factory creating the appropriate parser based on file extension."""

    @classmethod
    def get_parser(cls, file_path: Path | str) -> BaseDocumentParser:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return PDFParser()
        elif suffix in [".docx", ".doc"]:
            return DocxParser()
        elif suffix in [".pptx", ".ppt"]:
            return PptxParser()
        else:
            raise ValueError(f"Unsupported document format: '{suffix}'. Supported: PDF, DOCX, PPTX.")
