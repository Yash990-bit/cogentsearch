"""
Unit Tests for CogentSearch Pydantic V2 Schemas.
Verifies geometric invariants, coordinate normalization, and contract validation.
"""

import pytest
from cogent.schemas.document import (
    BoundingBox,
    VisualElement,
    VisualElementType,
    DocumentChunk,
    ChunkType,
    MultimodalDocument,
)


def test_bounding_box_valid():
    bbox = BoundingBox(page_number=1, x0=10.0, y0=20.0, x1=110.0, y1=120.0, page_width=595.0, page_height=842.0)
    assert bbox.width == 100.0
    assert bbox.height == 100.0
    assert bbox.area == 10000.0

    norm = bbox.to_normalized()
    assert 0.0 <= norm.x0 <= 1.0
    assert 0.0 <= norm.y0 <= 1.0
    assert 0.0 <= norm.x1 <= 1.0
    assert 0.0 <= norm.y1 <= 1.0


def test_bounding_box_invalid_inverted():
    with pytest.raises(ValueError, match="must be strictly greater than x0"):
        BoundingBox(page_number=1, x0=100.0, y0=20.0, x1=50.0, y1=120.0)

    with pytest.raises(ValueError, match="must be strictly greater than y0"):
        BoundingBox(page_number=1, x0=10.0, y0=120.0, x1=50.0, y1=20.0)


def test_bounding_box_out_of_bounds():
    with pytest.raises(ValueError, match="exceeds page width"):
        BoundingBox(page_number=1, x0=10.0, y0=20.0, x1=700.0, y1=120.0, page_width=595.0, page_height=842.0)


def test_visual_element_contract():
    bbox = BoundingBox(page_number=1, x0=10.0, y0=20.0, x1=100.0, y1=100.0)
    ve = VisualElement(
        element_id="elem_01",
        element_type=VisualElementType.TABLE,
        page_number=1,
        bounding_box=bbox,
        image_path="/tmp/test.png",
        extracted_text="Header 1 | Header 2\nVal 1 | Val 2",
        description="A benchmark comparison table",
        markdown_representation="| Header 1 | Header 2 |\n| --- | --- |\n| Val 1 | Val 2 |",
    )
    assert ve.extracted_text != ve.description
    assert ve.element_type == VisualElementType.TABLE


def test_multimodal_document_stats():
    bbox = BoundingBox(page_number=1, x0=10.0, y0=20.0, x1=100.0, y1=100.0)
    ve1 = VisualElement(element_id="e1", element_type=VisualElementType.TABLE, page_number=1, bounding_box=bbox, image_path="")
    ve2 = VisualElement(element_id="e2", element_type=VisualElementType.IMAGE, page_number=1, bounding_box=bbox, image_path="")
    chunk = DocumentChunk(chunk_id="c1", page_number=1, content="Test content", chunk_type=ChunkType.TEXT)

    doc = MultimodalDocument(
        document_id="doc_test",
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_type="pdf",
        total_pages=1,
        chunks=[chunk],
        visual_elements=[ve1, ve2],
    )
    stats = doc.get_summary_stats()
    assert stats["total_chunks"] == 1
    assert stats["total_visual_elements"] == 2
    assert stats["total_tables"] == 1
    assert stats["total_images"] == 1
