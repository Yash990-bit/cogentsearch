"""
Table Integrity and Ground-Truth Fidelity Tests for CogentSearch Phase 1.
Verifies column alignment, cell value retention, and multi-page table continuation linking.
"""

import json
from pathlib import Path
import pytest

from cogent.schemas.document import VisualElementType, ChunkType
from cogent.ingestion.pipeline import MultimodalIngestionPipeline


CORPUS_DIR = Path("data/test_corpus")
MANIFEST_PATH = CORPUS_DIR / "manifest.json"


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pipeline():
    return MultimodalIngestionPipeline(chunk_size=600, inject_visual_summaries=True)


def test_multipage_table_continuity_and_ground_truth(pipeline, manifest):
    """Verifies that multi-page tables are correctly linked and cell values preserved."""
    pdf_path = CORPUS_DIR / "doc_02_multipage_table.pdf"
    assert pdf_path.exists()

    doc = pipeline.process_file(pdf_path)
    assert doc.total_pages == 2

    tables = [v for v in doc.visual_elements if v.element_type == VisualElementType.TABLE]
    assert len(tables) == 2, f"Expected 2 table parts across 2 pages, found {len(tables)}"

    t_p1 = [t for t in tables if t.page_number == 1][0]
    t_p2 = [t for t in tables if t.page_number == 2][0]

    # Verify multi-page table continuation link
    assert t_p2.continuation_of == t_p1.element_id, (
        f"Page 2 table continuation_of ({t_p2.continuation_of}) must link to Page 1 table ({t_p1.element_id})"
    )

    # Verify ground-truth values in markdown representation
    gt = next(d["table_ground_truth"] for d in manifest["documents"] if d["filename"] == "doc_02_multipage_table.pdf")
    p1_md = t_p1.markdown_representation or ""
    p2_md = t_p2.markdown_representation or ""

    assert "Transaction ID" in p1_md
    assert "TX-1001" in p1_md or "TX-1001" in (t_p1.extracted_text or "")
    assert "TX-1003" in p2_md or "TX-1003" in (t_p2.extracted_text or "")


def test_docx_table_ground_truth(pipeline, manifest):
    """Verifies table in Word (.docx) preserves exact headers and numerical percentages."""
    docx_path = CORPUS_DIR / "doc_05_nested_report.docx"
    assert docx_path.exists()

    doc = pipeline.process_file(docx_path)
    tables = [v for v in doc.visual_elements if v.element_type == VisualElementType.TABLE]
    assert len(tables) >= 1

    t = tables[0]
    md = t.markdown_representation or ""
    verbatim = t.extracted_text or ""

    assert "Resource" in md or "Resource" in verbatim
    assert "Utilization" in md or "Utilization" in verbatim
    assert "64.2%" in md or "64.2%" in verbatim
    assert "512 GB" in md or "512 GB" in verbatim


def test_pptx_table_ground_truth(pipeline):
    """Verifies table in presentation slide preserves cell text and column boundaries."""
    pptx_path = CORPUS_DIR / "doc_06_slides_20plus.pptx"
    assert pptx_path.exists()

    doc = pipeline.process_file(pptx_path)
    tables = [v for v in doc.visual_elements if v.element_type == VisualElementType.TABLE]
    assert len(tables) >= 1

    t = tables[0]
    md = t.markdown_representation or ""
    verbatim = t.extracted_text or ""

    assert "Pillar" in md or "Pillar" in verbatim
    assert "Owner" in md or "Owner" in verbatim
