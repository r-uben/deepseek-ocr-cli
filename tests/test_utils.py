"""Tests for utility functions."""

import pytest
from pathlib import Path
from deepseek_ocr.utils import (
    is_supported_file,
    is_image_file,
    is_pdf_file,
    sanitize_filename,
    IMAGE_EXTENSIONS,
    PDF_EXTENSION,
)


class TestFileTypeChecking:
    """Tests for file type checking functions."""

    def test_is_supported_file_with_images(self) -> None:
        """Test supported file detection for images."""
        for ext in IMAGE_EXTENSIONS:
            assert is_supported_file(Path(f"test{ext}"))

    def test_is_supported_file_with_pdf(self) -> None:
        """Test supported file detection for PDF."""
        assert is_supported_file(Path("test.pdf"))
        assert is_supported_file(Path("test.PDF"))

    def test_is_supported_file_with_unsupported(self) -> None:
        """Test unsupported file detection."""
        assert not is_supported_file(Path("test.txt"))
        assert not is_supported_file(Path("test.docx"))
        assert not is_supported_file(Path("test.mp4"))

    def test_is_image_file(self) -> None:
        """Test image file detection."""
        assert is_image_file(Path("test.jpg"))
        assert is_image_file(Path("test.png"))
        assert not is_image_file(Path("test.pdf"))

    def test_is_pdf_file(self) -> None:
        """Test PDF file detection."""
        assert is_pdf_file(Path("test.pdf"))
        assert is_pdf_file(Path("test.PDF"))
        assert not is_pdf_file(Path("test.jpg"))


class TestFilenameSanitization:
    """Tests for filename sanitization."""

    def test_sanitize_removes_invalid_chars(self) -> None:
        """Test that invalid characters are replaced."""
        assert sanitize_filename('test<file>.pdf') == 'test_file_.pdf'
        assert sanitize_filename('test/file\\name.pdf') == 'test_file_name.pdf'
        assert sanitize_filename('test:file|name.pdf') == 'test_file_name.pdf'

    def test_sanitize_handles_empty(self) -> None:
        """Test that empty filenames get a default."""
        assert sanitize_filename('') == 'untitled'
        assert sanitize_filename('   ') == 'untitled'

    def test_sanitize_strips_spaces_and_dots(self) -> None:
        """Test that leading/trailing spaces and dots are removed."""
        assert sanitize_filename(' test.pdf ') == 'test.pdf'
        assert sanitize_filename('.test.pdf.') == 'test.pdf'

    def test_sanitize_preserves_valid_names(self) -> None:
        """Test that valid filenames are preserved."""
        assert sanitize_filename('valid_filename.pdf') == 'valid_filename.pdf'
        assert sanitize_filename('test-file-123.png') == 'test-file-123.png'
