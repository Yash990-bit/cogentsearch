"""
Universal Document Parser Module for CogentSearch.
Provides specialized parsers for PDF, DOCX (Word), and PPTX (PowerPoint) files,
preserving layout hierarchy, heading trees, multi-column reading order, page/slide numbers,
and adversarial exception handling.
"""

from __future__ import annotations
import io
import os
import zipfile
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import fitz  # PyMuPDF
from PIL import Image

from cogent.core.config import settings
from cogent.core.exceptions import (
    EmptyDocumentError,
    CorruptedDocumentError,
    PasswordProtectedError,
    UnsupportedFormatError,
)
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
    """High-fidelity PDF parser with reading-order sorting, multi-column support, and adversarial defenses."""

    def __init__(self, vision_extractor: Optional[VisionExtractor] = None):
        self.vision_extractor = vision_extractor or VisionExtractor()

    def parse(self, file_path: Path | str) -> RawParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise EmptyDocumentError(f"PDF file '{path.name}' is 0 bytes (empty).", filename=path.name)

        file_bytes = path.read_bytes()

        # Check for PDF header signature '%PDF'
        if not file_bytes.startswith(b"%PDF"):
            raise CorruptedDocumentError(
                f"File '{path.name}' lacks valid PDF magic bytes.", filename=path.name
            )

        # Content-based hash ensures duplicate filenames with different contents have unique IDs
        doc_id = hashlib.sha256(file_bytes[:65536] + str(file_size).encode() + file_bytes[-4096:]).hexdigest()[:12]

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise CorruptedDocumentError(
                f"Failed to open PDF '{path.name}': {e}", filename=path.name, details={"error": str(e)}
            )

        if doc.is_encrypted or doc.needs_pass:
            doc.close()
            raise PasswordProtectedError(
                f"PDF '{path.name}' is encrypted or password-protected.", filename=path.name
            )

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise EmptyDocumentError(f"PDF '{path.name}' contains zero pages.", filename=path.name)

        page_texts: Dict[int, str] = {}
        page_headers: Dict[int, List[str]] = {}
        all_visual_elements: List[VisualElement] = []

        logger.info(f"Parsing PDF '{path.name}' ({total_pages} pages)...")

        body_font_size = self._estimate_body_font_size(doc)
        current_hierarchy: List[str] = []
        previous_tables: List[VisualElement] = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_w = page.rect.width
            page_h = page.rect.height

            # Check if multi-column layout is present and sort blocks accordingly
            sorted_blocks = self._extract_sorted_text_blocks(page, page_w)

            page_text_lines: List[str] = []
            page_detected_headers: List[str] = []

            for block in sorted_blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        line_text = "".join(s.get("text", "") for s in spans).strip()
                        if not line_text:
                            continue

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

            # Scanned page check: images present but no digital text layer
            is_scanned = len(combined_page_text) == 0 and len(page.get_images()) > 0
            if is_scanned:
                logger.warning(
                    f"Page {page_num} in '{path.name}' appears to be a scanned image with no embedded text layer. "
                    "Extracting visual assets and transcribing via Vision LLM."
                )

            page_texts[page_num] = combined_page_text
            page_headers[page_num] = list(current_hierarchy)

            # 2. Extract visual elements with multi-page table tracking
            page_visuals = self.vision_extractor.extract_from_pdf_page(
                doc_id=doc_id,
                page=page,
                page_number=page_num,
                page_text_context=combined_page_text,
                previous_page_tables=previous_tables,
            )
            all_visual_elements.extend(page_visuals)

            # Update previous tables for multi-page table linking
            previous_tables = [v for v in page_visuals if v.element_type == VisualElementType.TABLE]

        metadata = dict(doc.metadata) if doc.metadata else {}
        metadata["rotation_detected"] = any(doc[p].rotation != 0 for p in range(total_pages))
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
    def _extract_sorted_text_blocks(page: fitz.Page, page_w: float) -> List[Dict[str, Any]]:
        """Sorts text blocks in visual reading order (left-to-right columns, top-to-bottom)."""
        raw_blocks = page.get_text("dict")["blocks"]
        text_blocks = [b for b in raw_blocks if b.get("type") == 0]

        if not text_blocks:
            return []

        # Check if page is two-column: blocks distributed on left half and right half
        mid_x = page_w / 2.0
        left_blocks = [b for b in text_blocks if b["bbox"][2] <= mid_x + 30.0]
        right_blocks = [b for b in text_blocks if b["bbox"][0] >= mid_x - 30.0]

        # If significant proportion of blocks clearly separate into left/right columns
        if len(left_blocks) >= 2 and len(right_blocks) >= 2 and (len(left_blocks) + len(right_blocks)) >= len(text_blocks) * 0.7:
            # Sort left column top-to-bottom, then right column top-to-bottom
            left_blocks.sort(key=lambda b: b["bbox"][1])
            right_blocks.sort(key=lambda b: b["bbox"][1])
            remaining = [b for b in text_blocks if b not in left_blocks and b not in right_blocks]
            remaining.sort(key=lambda b: b["bbox"][1])
            return left_blocks + right_blocks + remaining
        else:
            # Standard single column or top-to-bottom reading order
            text_blocks.sort(key=lambda b: b["bbox"][1])
            return text_blocks

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
        return max(set(sizes), key=sizes.count)


class DocxParser(BaseDocumentParser):
    """Parser for Microsoft Word (.docx) documents with corrupted file defense."""

    def __init__(self, vision_client: Optional[VisionLLMClient] = None):
        self.vision_client = vision_client or VisionLLMClient()
        self.output_dir = settings.EXTRACTED_IMAGES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, file_path: Path | str) -> RawParsedDocument:
        import docx

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise EmptyDocumentError(f"Word document '{path.name}' is 0 bytes (empty).", filename=path.name)

        file_bytes = path.read_bytes()
        doc_id = hashlib.sha256(file_bytes[:65536] + str(file_size).encode() + file_bytes[-4096:]).hexdigest()[:12]

        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except (zipfile.BadZipFile, Exception) as e:
            raise CorruptedDocumentError(
                f"Failed to open DOCX '{path.name}': {e}", filename=path.name, details={"error": str(e)}
            )

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
                verbatim_text = " | ".join(" ".join(r) for r in table_rows)
                text_lines.append(f"\n{md_table}\n")

                visual_elements.append(
                    VisualElement(
                        element_id=elem_id,
                        element_type=VisualElementType.TABLE,
                        page_number=1,
                        bounding_box=BoundingBox(page_number=1, x0=10.0, y0=10.0, x1=500.0, y1=200.0, page_width=612.0, page_height=792.0),
                        image_path="",
                        description=f"Extracted Word Document Table: {elem_id}",
                        extracted_text=verbatim_text,
                        markdown_representation=md_table,
                        confidence=1.0,
                    )
                )

        # 3. Extract embedded media/images
        img_idx = 0
        try:
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
                            bounding_box=BoundingBox(page_number=1, x0=10.0, y0=220.0, x1=400.0, y1=450.0, page_width=612.0, page_height=792.0),
                            image_path=str(img_path),
                            description=llm_res["description"],
                            confidence=0.90,
                        )
                    )
                    img_idx += 1
        except Exception as e:
            logger.warning(f"Failed extracting media parts from DOCX: {e}")

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
            raw_metadata={},
        )


class PptxParser(BaseDocumentParser):
    """Parser for PowerPoint (.pptx) presentations with corrupted file defense."""

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

        file_size = path.stat().st_size
        if file_size == 0:
            raise EmptyDocumentError(f"PowerPoint presentation '{path.name}' is 0 bytes (empty).", filename=path.name)

        file_bytes = path.read_bytes()
        doc_id = hashlib.sha256(file_bytes[:65536] + str(file_size).encode() + file_bytes[-4096:]).hexdigest()[:12]

        try:
            prs = Presentation(io.BytesIO(file_bytes))
        except (zipfile.BadZipFile, Exception) as e:
            raise CorruptedDocumentError(
                f"Failed to open PPTX '{path.name}': {e}", filename=path.name, details={"error": str(e)}
            )

        total_slides = len(prs.slides)
        if total_slides == 0:
            raise EmptyDocumentError(f"Presentation '{path.name}' contains zero slides.", filename=path.name)

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

            if slide.shapes.title and slide.shapes.title.text:
                slide_title = slide.shapes.title.text.strip()
                slide_lines.append(f"# Slide {slide_num}: {slide_title}\n")

            img_idx = 0
            for shape in slide.shapes:
                x0 = max(0.0, shape.left.pt if shape.left else 0.0)
                y0 = max(0.0, shape.top.pt if shape.top else 0.0)
                w = max(10.0, shape.width.pt if shape.width else 10.0)
                h = max(10.0, shape.height.pt if shape.height else 10.0)
                x1 = min(slide_width, x0 + w)
                y1 = min(slide_height, y0 + h)

                bbox = BoundingBox(
                    page_number=slide_num,
                    x0=round(x0, 2),
                    y0=round(y0, 2),
                    x1=round(x1, 2),
                    y1=round(y1, 2),
                    page_width=slide_width,
                    page_height=slide_height,
                )

                if shape.has_text_frame and shape != slide.shapes.title:
                    for p in shape.text_frame.paragraphs:
                        p_text = p.text.strip()
                        if p_text:
                            slide_lines.append(f"- {p_text}")

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
                        verbatim_text = " | ".join(" ".join(r) for r in rows_data)
                        slide_lines.append(f"\n{md_table}\n")

                        visual_elements.append(
                            VisualElement(
                                element_id=elem_id,
                                element_type=VisualElementType.TABLE,
                                page_number=slide_num,
                                bounding_box=bbox,
                                image_path="",
                                description=f"Presentation Table on Slide {slide_num}: {slide_title}",
                                extracted_text=verbatim_text,
                                markdown_representation=md_table,
                                confidence=1.0,
                            )
                        )
                    img_idx += 1

                elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    elem_id = f"img_{doc_id}_s{slide_num}_{img_idx}"
                    try:
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
                    except Exception as e:
                        logger.warning(f"Failed extracting picture shape on slide {slide_num}: {e}")
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


class LlamaParseAdapter(BaseDocumentParser):
    """
    Provider-agnostic adapter for LlamaParse (LlamaCloud multimodal parser).
    Converts LlamaParse structured layout blocks, tables, and bounding boxes into RawParsedDocument.
    Falls back gracefully to native high-fidelity parser when offline or unconfigured.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, "LLAMA_CLOUD_API_KEY", "")

    def parse(self, file_path: Path | str) -> RawParsedDocument:
        path = Path(file_path)
        try:
            if self.api_key:
                from llama_parse import LlamaParse
                logger.info(f"Invoking LlamaParse provider for: {path.name}")
                # Real cloud parsing path when API key is present
                parser = LlamaParse(api_key=self.api_key, result_type="markdown")
                parsed_docs = parser.load_data(str(path))
                if parsed_docs:
                    page_texts = {i + 1: doc.text for i, doc in enumerate(parsed_docs)}
                    return RawParsedDocument(
                        document_id=hashlib.md5(path.name.encode()).hexdigest()[:12],
                        filename=path.name,
                        file_path=str(path.resolve()),
                        file_type=path.suffix.lstrip(".").lower(),
                        total_pages=len(parsed_docs),
                        page_texts=page_texts,
                        page_headers={},
                        visual_elements=[],
                        raw_metadata={"provider": "llamaparse"},
                    )
        except Exception as exc:
            logger.warning(f"LlamaParse provider unavailable ({exc}); falling back to native engine.")
        return PDFParser().parse(path)


class DoclingAdapter(BaseDocumentParser):
    """
    Provider-agnostic adapter for IBM Docling document conversion.
    Converts Docling layout models and reading order into RawParsedDocument.
    Falls back gracefully to native high-fidelity parser when offline.
    """

    def parse(self, file_path: Path | str) -> RawParsedDocument:
        path = Path(file_path)
        try:
            from docling.document_converter import DocumentConverter
            logger.info(f"Invoking Docling converter for: {path.name}")
            converter = DocumentConverter()
            result = converter.convert(path)
            doc = result.document
            markdown_content = doc.export_to_markdown()
            return RawParsedDocument(
                document_id=hashlib.md5(path.name.encode()).hexdigest()[:12],
                filename=path.name,
                file_path=str(path.resolve()),
                file_type=path.suffix.lstrip(".").lower(),
                total_pages=1,
                page_texts={1: markdown_content},
                page_headers={},
                visual_elements=[],
                raw_metadata={"provider": "docling"},
            )
        except Exception as exc:
            logger.debug(f"Docling provider fallback to native engine: {exc}")
        return PDFParser().parse(path)


class UniversalParserFactory:
    """
    Provider-agnostic parser factory creating format and provider-specific parsers.
    Supports Docling, LlamaParse, and native PyMuPDF/python-docx/python-pptx engines.
    """

    @classmethod
    def get_parser(
        cls,
        file_path: Path | str,
        provider: Optional[str] = None,
    ) -> BaseDocumentParser:
        if provider == "llamaparse":
            return LlamaParseAdapter()
        elif provider == "docling":
            return DoclingAdapter()

        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return PDFParser()
        elif suffix in [".docx", ".doc"]:
            return DocxParser()
        elif suffix in [".pptx", ".ppt"]:
            return PptxParser()
        else:
            raise UnsupportedFormatError(
                f"Unsupported document format '{suffix}' for file '{Path(file_path).name}'. "
                f"Supported formats: .pdf, .docx, .pptx",
                filename=Path(file_path).name,
            )
