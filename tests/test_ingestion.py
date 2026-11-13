"""
Automated Integration and Unit Test Suite for CogentSearch Phase 1.
Generates synthetic PDF, DOCX, and PPTX documents with real visual assets
and validates end-to-end multimodal parsing, vision extraction, and chunk enrichment.
"""

import pytest
from pathlib import Path
import fitz
from PIL import Image, ImageDraw
import docx
from pptx import Presentation
from pptx.util import Inches, Pt

from cogent.schemas.document import (
    MultimodalDocument,
    VisualElementType,
    ChunkType,
)
from cogent.ingestion.pipeline import MultimodalIngestionPipeline


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory) -> Path:
    """Creates a temporary directory with synthetic test documents."""
    temp_dir = tmp_path_factory.mktemp("test_docs")

    # 1. Create a dummy test image
    img_path = temp_dir / "sample_chart.png"
    img = Image.new("RGB", (300, 200), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Test Chart: Revenue Growth", fill=(255, 255, 0))
    d.rectangle([50, 50, 250, 150], outline=(255, 255, 255), width=2)
    img.save(img_path)

    # 2. Generate a synthetic PDF with text, headings, table, and image
    pdf_path = temp_dir / "sample_multimodal.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 50), "Chapter 1: System Overview", fontsize=20, fontname="helv")
    page1.insert_text((50, 80), "This chapter outlines the foundational principles of multimodal AI architectures.", fontsize=11)
    page1.insert_text((50, 110), "1.1 Architectural Blueprint", fontsize=15, fontname="helv")
    page1.insert_text((50, 130), "The system decouples visual extraction from dense semantic retrieval.", fontsize=11)

    # Insert image on Page 1
    rect = fitz.Rect(50, 180, 350, 380)
    page1.insert_image(rect, filename=str(img_path))

    # Page 2 (with Table)
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 50), "Chapter 2: Quantitative Benchmarks", fontsize=20, fontname="helv")
    # Draw simple table lines
    page2.draw_rect(fitz.Rect(50, 100, 450, 200), color=(0, 0, 0), width=1)
    page2.draw_line(fitz.Point(50, 130), fitz.Point(450, 130), color=(0, 0, 0), width=1)
    page2.draw_line(fitz.Point(200, 100), fitz.Point(200, 200), color=(0, 0, 0), width=1)
    page2.insert_text((60, 120), "Model Name", fontsize=12, fontname="helv")
    page2.insert_text((220, 120), "Precision Score", fontsize=12, fontname="helv")
    page2.insert_text((60, 160), "CogentSearch-v1", fontsize=11)
    page2.insert_text((220, 160), "94.8%", fontsize=11)

    doc.save(str(pdf_path))
    doc.close()

    # 3. Generate a synthetic DOCX with heading, paragraph, table, and image
    docx_path = temp_dir / "sample_report.docx"
    doc_docx = docx.Document()
    doc_docx.add_heading("Executive Summary", level=1)
    doc_docx.add_paragraph("This document details the multi-agent search engine evaluation.")
    doc_docx.add_heading("Key Findings", level=2)
    doc_docx.add_paragraph("Retrieval recall improved by 35% with visual injection.")

    # Table
    table = doc_docx.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Category"
    table.cell(0, 1).text = "Status"
    table.cell(1, 0).text = "Vision Ingestion"
    table.cell(1, 1).text = "Production Ready"

    # Add image
    doc_docx.add_picture(str(img_path), width=docx.shared.Inches(2.5))
    doc_docx.save(str(docx_path))

    # 4. Generate a synthetic PPTX
    pptx_path = temp_dir / "sample_deck.pptx"
    prs = Presentation()
    slide_layout = prs.slide_layouts[1]  # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "CogentSearch Architecture"
    content = slide.placeholders[1]
    content.text = "Next-Generation Multimodal Agentic Retrieval\nPreserves page hierarchy and bounding boxes"

    # Add table to slide
    rows, cols, left, top, width, height = 2, 2, Inches(2), Inches(3.5), Inches(4), Inches(1)
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table_shape.table.cell(0, 0).text = "Feature"
    table_shape.table.cell(0, 1).text = "Benchmark"
    table_shape.table.cell(1, 0).text = "Latency"
    table_shape.table.cell(1, 1).text = "< 120ms"

    # Add image to slide
    slide.shapes.add_picture(str(img_path), Inches(6.5), Inches(3.5), width=Inches(2.5))
    prs.save(str(pptx_path))

    return temp_dir


def test_pdf_multimodal_ingestion(sample_data_dir: Path):
    """Verifies that PDF parsing captures text, headers, images, and injects visual summaries."""
    pdf_path = sample_data_dir / "sample_multimodal.pdf"
    pipeline = MultimodalIngestionPipeline(chunk_size=400, inject_visual_summaries=True)

    result = pipeline.process_file(pdf_path)

    assert isinstance(result, MultimodalDocument)
    assert result.total_pages == 2
    assert result.file_type == "pdf"
    assert len(result.chunks) > 0

    # Ensure visual elements were extracted
    assert len(result.visual_elements) >= 1
    visual_types = [v.element_type for v in result.visual_elements]
    assert VisualElementType.IMAGE in visual_types or VisualElementType.TABLE in visual_types

    # Ensure visual summaries are injected into document chunks
    visual_chunks = [c for c in result.chunks if c.chunk_type in [ChunkType.VISUAL_SUMMARY, ChunkType.TABLE]]
    assert len(visual_chunks) > 0
    assert any("Bounding Box Coordinates" in c.content for c in visual_chunks)


def test_docx_ingestion(sample_data_dir: Path):
    """Verifies Word docx ingestion, table extraction, and image extraction."""
    docx_path = sample_data_dir / "sample_report.docx"
    pipeline = MultimodalIngestionPipeline(chunk_size=400, inject_visual_summaries=True)

    result = pipeline.process_file(docx_path)

    assert isinstance(result, MultimodalDocument)
    assert result.file_type == "docx"
    assert len(result.chunks) > 0

    # Verify table representation is extracted
    has_table = any(v.element_type == VisualElementType.TABLE for v in result.visual_elements)
    assert has_table


def test_pptx_ingestion(sample_data_dir: Path):
    """Verifies PowerPoint pptx ingestion, slide titles, tables, and images."""
    pptx_path = sample_data_dir / "sample_deck.pptx"
    pipeline = MultimodalIngestionPipeline(chunk_size=400, inject_visual_summaries=True)

    result = pipeline.process_file(pptx_path)

    assert isinstance(result, MultimodalDocument)
    assert result.file_type == "pptx"
    assert result.total_pages == 1
    assert len(result.chunks) > 0

    # Slide titles must be recognized as headers
    header_found = any("CogentSearch Architecture" in h for c in result.chunks for h in c.headers_hierarchy)
    assert header_found


def test_complete_multimodal_acceptance_criterion():
    """
    Acceptance Criterion:
    A real document containing text, table, and image must be successfully ingested:
    - Text blocks & headers accurately mapped
    - Table detected, cropped, and represented in structured markdown
    - Image detected, cropped, and annotated with semantic description
    - Both injected into the document retrieval schema alongside exact bounding box coordinates
    """
    report_pdf = Path("data/uploads/sample_enterprise_report.pdf")
    assert report_pdf.exists(), "Sample enterprise report PDF must exist"

    pipeline = MultimodalIngestionPipeline(chunk_size=400, inject_visual_summaries=True)
    result = pipeline.process_file(report_pdf)

    # 1. Structural assertions
    assert isinstance(result, MultimodalDocument)
    assert result.total_pages == 2
    assert result.file_type == "pdf"

    # 2. Multimodal asset assertions: BOTH table AND image must be present
    extracted_types = {v.element_type for v in result.visual_elements}
    assert VisualElementType.IMAGE in extracted_types, "Must extract embedded image"
    assert VisualElementType.TABLE in extracted_types, "Must extract embedded table"

    # 3. Check cropped asset files exist on disk
    for elem in result.visual_elements:
        crop_path = Path(elem.image_path)
        assert crop_path.exists(), f"Cropped image asset {crop_path} must exist on disk"
        assert crop_path.stat().st_size > 0, f"Cropped image {crop_path} must not be empty"

    # 4. Schema injection assertions: chunks must contain visual summaries and bounding boxes
    table_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.TABLE]
    image_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.VISUAL_SUMMARY]
    text_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.TEXT]

    assert len(table_chunks) >= 1, "Must have at least one injected table chunk"
    assert len(image_chunks) >= 1, "Must have at least one injected image chunk"
    assert len(text_chunks) >= 1, "Must have textual body chunks"

    # 5. Provenance validation
    for v_chunk in table_chunks + image_chunks:
        assert len(v_chunk.bounding_boxes) > 0, "Visual chunk must maintain bounding box coordinates"
        bbox = v_chunk.bounding_boxes[0]
        assert bbox.page_number in [1, 2]
        assert bbox.width > 0
        assert bbox.height > 0


