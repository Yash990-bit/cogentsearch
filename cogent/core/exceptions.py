"""
Domain-Specific Exceptions for CogentSearch Ingestion Pipeline.
Enforces strict error categorization for adversarial edge cases.
"""

from typing import Optional, Dict, Any


class DocumentIngestionError(Exception):
    """Base exception for all document ingestion failures."""

    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.filename = filename
        self.details = details or {}


class EmptyDocumentError(DocumentIngestionError):
    """Raised when an uploaded file is 0 bytes or completely empty."""
    pass


class CorruptedDocumentError(DocumentIngestionError):
    """Raised when a file has a valid extension but malformed or corrupted internal binary structure."""
    pass


class PasswordProtectedError(DocumentIngestionError):
    """Raised when an encrypted or password-protected document is submitted without credentials."""
    pass


class UnsupportedFormatError(DocumentIngestionError):
    """Raised when a file format is not supported by the ingestion engine."""
    pass


class OCRUnavailableWarning(UserWarning):
    """Warned when a scanned page is detected but system OCR is unavailable on PATH."""
    pass
