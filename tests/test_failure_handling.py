"""
Adversarial and Failure-Handling Tests for CogentSearch Phase 1.
Validates robust rejection and graceful failure handling across corrupted, encrypted,
empty, and unsupported document payloads.
"""

import pytest
from pathlib import Path
from cogent.core.exceptions import (
    EmptyDocumentError,
    CorruptedDocumentError,
    PasswordProtectedError,
    UnsupportedFormatError,
)
from cogent.ingestion.pipeline import MultimodalIngestionPipeline


CORPUS_DIR = Path("data/test_corpus")


@pytest.fixture
def pipeline():
    return MultimodalIngestionPipeline(chunk_size=500, inject_visual_summaries=True)


def test_empty_document_rejection(pipeline):
    empty_pdf = CORPUS_DIR / "adv_empty.pdf"
    assert empty_pdf.exists()
    with pytest.raises(EmptyDocumentError, match="is 0 bytes"):
        pipeline.process_file(empty_pdf)


def test_corrupted_document_rejection(pipeline):
    corrupted_pdf = CORPUS_DIR / "adv_corrupted.pdf"
    assert corrupted_pdf.exists()
    with pytest.raises(CorruptedDocumentError):
        pipeline.process_file(corrupted_pdf)


def test_encrypted_document_rejection(pipeline):
    encrypted_pdf = CORPUS_DIR / "adv_encrypted.pdf"
    assert encrypted_pdf.exists()
    with pytest.raises(PasswordProtectedError, match="encrypted or password-protected"):
        pipeline.process_file(encrypted_pdf)


def test_unsupported_format_rejection(pipeline):
    unsupp = CORPUS_DIR / "adv_unsupported.xyz"
    assert unsupp.exists()
    with pytest.raises(UnsupportedFormatError, match="Unsupported document format"):
        pipeline.process_file(unsupp)


def test_no_text_pdf_graceful_handling(pipeline):
    no_text_pdf = CORPUS_DIR / "adv_no_text.pdf"
    assert no_text_pdf.exists()
    doc = pipeline.process_file(no_text_pdf)
    assert doc.total_pages == 1
    # Document has 0 text chunks, doesn't crash
    assert len(doc.chunks) == 0


def test_duplicate_filename_content_hash_idempotency(pipeline, tmp_path):
    # Two files with identical filename but different contents
    f1 = tmp_path / "report.pdf"
    f2 = tmp_path / "subfolder" / "report.pdf"
    f2.parent.mkdir(parents=True, exist_ok=True)

    import fitz
    d1 = fitz.open()
    d1.new_page().insert_text((50, 50), "Content Version Alpha")
    d1.save(str(f1))
    d1.close()

    d2 = fitz.open()
    d2.new_page().insert_text((50, 50), "Content Version Beta with distinct payload")
    d2.save(str(f2))
    d2.close()

    res1 = pipeline.process_file(f1)
    res2 = pipeline.process_file(f2)

    # Different contents must have different IDs despite identical filename
    assert res1.document_id != res2.document_id

    # Repeated upload of f1 must produce identical deterministic ID
    res1_repeat = pipeline.process_file(f1)
    assert res1.document_id == res1_repeat.document_id
