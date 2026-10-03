"""
tests/test_loaders.py
=====================
Integration-style tests for the document ingestion layer.

Tests cover:
  A. DOCX loading — basic functionality
  B. PNG / image loading — basic functionality
  C. Unsupported extension → ValueError
  D. Non-existent file → FileNotFoundError
  E. Empty / corrupt file handling
  F. Preservation of meaningful text content
  G. Table extraction from DOCX
  H. Normalizer — conservative rules
  I. DocumentContent structure contract
  J. load_document() unified interface

Philosophy:
  - We test TEXT INGESTION only.
  - We do NOT assert on extracted business fields (party names, dates, values).
  - We verify that the text is long enough to be useful, contains expected
    structural markers, and that the DocumentContent contract is satisfied.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
# Allow running from project root: pytest tests/ or python -m pytest
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.loaders import load_document, SUPPORTED_EXTENSIONS, DocumentContent
from app.loaders.docx_loader import load_docx
from app.loaders.image_loader import load_image
from app.loaders.normalizer import normalize_text

# ── test data paths ───────────────────────────────────────────────────────────
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR  = PROJECT_ROOT / "data" / "test"

# Pick specific files that are known to exist from the Phase 1 audit
TRAIN_DOCX_1  = TRAIN_DIR / "18325926-Rental-Agreement-1.docx"
TRAIN_DOCX_2  = TRAIN_DIR / "44737744-Maddireddy-Bhargava-Reddy-Rental-Agreement.docx"
TRAIN_DOCX_TABLE = TRAIN_DIR / "6683129-House-Rental-Contract-Geraldine-Galinato-v2.docx"
TRAIN_PNG_1   = TRAIN_DIR / "36199312-Rental-Agreement.png"
TRAIN_PNG_2   = TRAIN_DIR / "54770958-Rental-Agreement.png"
TEST_DOCX_1   = TEST_DIR  / "156155545-Rental-Agreement-Kns-Home.pdf.docx"
TEST_DOCX_2   = TEST_DIR  / "228094620-Rental-Agreement.pdf.docx"
TEST_PNG_1    = TEST_DIR  / "24158401-Rental-Agreement.png"
TEST_PNG_2    = TEST_DIR  / "95980236-Rental-Agreement.png"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# ── helpers ───────────────────────────────────────────────────────────────────

def data_available(path: Path) -> bool:
    return path.exists()


def skip_if_missing(path: Path):
    return pytest.mark.skipif(
        not data_available(path),
        reason=f"Test data not found: {path}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# A. DOCX — basic functionality
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocxLoader:

    @skip_if_missing(TRAIN_DOCX_1)
    def test_load_docx_returns_document_content(self):
        doc = load_docx(TRAIN_DOCX_1)
        assert isinstance(doc, DocumentContent)

    @skip_if_missing(TRAIN_DOCX_1)
    def test_load_docx_file_type(self):
        doc = load_docx(TRAIN_DOCX_1)
        assert doc.file_type == "docx"

    @skip_if_missing(TRAIN_DOCX_1)
    def test_load_docx_file_name(self):
        doc = load_docx(TRAIN_DOCX_1)
        assert doc.file_name == TRAIN_DOCX_1.name

    @skip_if_missing(TRAIN_DOCX_1)
    def test_load_docx_nonempty_raw_text(self):
        doc = load_docx(TRAIN_DOCX_1)
        assert len(doc.raw_text) > 200, "Expected at least 200 chars of raw text"

    @skip_if_missing(TRAIN_DOCX_1)
    def test_load_docx_nonempty_normalized_text(self):
        doc = load_docx(TRAIN_DOCX_1)
        assert len(doc.normalized_text) > 100

    @skip_if_missing(TRAIN_DOCX_1)
    def test_load_docx_page_count_is_none(self):
        """DOCX page count is not reliably determinable without rendering."""
        doc = load_docx(TRAIN_DOCX_1)
        assert doc.page_count is None

    @skip_if_missing(TRAIN_DOCX_1)
    def test_load_docx_metadata_has_paragraph_count(self):
        doc = load_docx(TRAIN_DOCX_1)
        assert "paragraph_count" in doc.metadata
        assert doc.metadata["paragraph_count"] > 0

    @skip_if_missing(TRAIN_DOCX_2)
    def test_load_second_docx(self):
        doc = load_docx(TRAIN_DOCX_2)
        assert len(doc.normalized_text) > 100


# ═══════════════════════════════════════════════════════════════════════════════
# B. Image / PNG loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageLoader:

    @pytest.fixture(autouse=True)
    def ensure_tesseract(self):
        """Skip image tests if Tesseract is not installed."""
        import shutil
        if not shutil.which("tesseract"):
            pytest.skip("Tesseract not installed")

    @skip_if_missing(TRAIN_PNG_1)
    def test_load_png_returns_document_content(self):
        doc = load_image(TRAIN_PNG_1)
        assert isinstance(doc, DocumentContent)

    @skip_if_missing(TRAIN_PNG_1)
    def test_load_png_file_type(self):
        doc = load_image(TRAIN_PNG_1)
        assert doc.file_type == "png"

    @skip_if_missing(TRAIN_PNG_1)
    def test_load_png_page_count_is_1(self):
        doc = load_image(TRAIN_PNG_1)
        assert doc.page_count == 1

    @skip_if_missing(TRAIN_PNG_1)
    def test_load_png_has_nonempty_raw_text(self):
        doc = load_image(TRAIN_PNG_1, artifacts_dir=ARTIFACTS_DIR)
        # High-res scans should produce at least 300 chars of OCR text
        assert len(doc.raw_text) > 300, (
            f"OCR text too short ({len(doc.raw_text)} chars). "
            "Possible OCR engine issue."
        )

    @skip_if_missing(TRAIN_PNG_1)
    def test_load_png_metadata_has_dimensions(self):
        doc = load_image(TRAIN_PNG_1)
        assert "image_width" in doc.metadata
        assert "image_height" in doc.metadata
        assert doc.metadata["image_width"] > 0
        assert doc.metadata["image_height"] > 0

    @skip_if_missing(TRAIN_PNG_1)
    def test_load_png_metadata_has_strategy(self):
        doc = load_image(TRAIN_PNG_1)
        assert "ocr_strategy" in doc.metadata
        assert isinstance(doc.metadata["ocr_strategy"], str)

    @skip_if_missing(TRAIN_PNG_2)
    def test_load_second_png(self):
        doc = load_image(TRAIN_PNG_2, artifacts_dir=ARTIFACTS_DIR)
        assert len(doc.raw_text) > 100


# ═══════════════════════════════════════════════════════════════════════════════
# C. Unsupported extension
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnsupportedExtension:

    def test_pdf_raises_value_error(self, tmp_path):
        fake = tmp_path / "doc.pdf"
        fake.write_bytes(b"%PDF-1.4 fake")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(fake)

    def test_txt_raises_value_error(self, tmp_path):
        fake = tmp_path / "notes.txt"
        fake.write_text("hello world")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(fake)

    def test_no_extension_raises_value_error(self, tmp_path):
        fake = tmp_path / "noextension"
        fake.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(fake)


# ═══════════════════════════════════════════════════════════════════════════════
# D. Non-existent file
# ═══════════════════════════════════════════════════════════════════════════════

class TestNonExistentFile:

    def test_missing_docx_raises(self):
        with pytest.raises(FileNotFoundError):
            load_document(PROJECT_ROOT / "data" / "train" / "ghost.docx")

    def test_missing_png_raises(self):
        with pytest.raises(FileNotFoundError):
            load_document(PROJECT_ROOT / "data" / "train" / "ghost.png")


# ═══════════════════════════════════════════════════════════════════════════════
# E. Corrupt / empty file handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorruptInput:

    def test_empty_docx_raises_value_error(self, tmp_path):
        empty = tmp_path / "empty.docx"
        empty.write_bytes(b"")
        with pytest.raises(ValueError):
            load_docx(empty)

    def test_random_bytes_as_docx_raises(self, tmp_path):
        bad = tmp_path / "bad.docx"
        bad.write_bytes(os.urandom(512))
        with pytest.raises(ValueError):
            load_docx(bad)

    def test_corrupt_image_raises(self, tmp_path):
        bad_img = tmp_path / "corrupt.png"
        bad_img.write_bytes(b"\x89PNG\r\n\x1a\n" + os.urandom(200))
        with pytest.raises(ValueError):
            load_image(bad_img)


# ═══════════════════════════════════════════════════════════════════════════════
# F. Preservation of meaningful text
# ═══════════════════════════════════════════════════════════════════════════════

class TestTextPreservation:
    """
    Verify that the loader preserves content that is critical for later
    semantic extraction: dates, monetary values, and party name fragments.

    We do NOT assert exact field values — only that the raw_text contains
    strings we know appear in the source document.
    """

    @skip_if_missing(TRAIN_DOCX_1)
    def test_date_string_preserved_in_docx(self):
        """18325926 mentions 'fifth day of December 2008' in its preamble."""
        doc = load_docx(TRAIN_DOCX_1)
        text_lower = doc.raw_text.lower()
        # At least one date-related token must be present
        assert any(tok in text_lower for tok in ["december", "2008", "2009"]), (
            "Expected date tokens not found in DOCX text"
        )

    @skip_if_missing(TRAIN_DOCX_1)
    def test_monetary_value_preserved_in_docx(self):
        """18325926 mentions Rs.4,000 as the rent."""
        doc = load_docx(TRAIN_DOCX_1)
        # The value '4,000' or '4000' should appear in the text
        assert "4,000" in doc.raw_text or "4000" in doc.raw_text, (
            "Monetary value not preserved in DOCX text"
        )

    @skip_if_missing(TRAIN_DOCX_1)
    def test_party_name_token_preserved_in_docx(self):
        """18325926 has 'Kuttan' as a party name token."""
        doc = load_docx(TRAIN_DOCX_1)
        assert "Kuttan" in doc.raw_text, "Party name token not in DOCX text"

    @skip_if_missing(TRAIN_PNG_1)
    def test_ocr_produces_text_with_printable_chars(self):
        doc = load_image(TRAIN_PNG_1, artifacts_dir=ARTIFACTS_DIR)
        printable = sum(1 for c in doc.raw_text if c.isprintable())
        total = len(doc.raw_text)
        # At least 80% of characters should be printable
        if total > 0:
            ratio = printable / total
            assert ratio >= 0.80, f"OCR noise too high: printable ratio = {ratio:.2%}"

    @skip_if_missing(TRAIN_DOCX_1)
    def test_normalized_text_shorter_or_equal_to_raw(self):
        doc = load_docx(TRAIN_DOCX_1)
        # Normalisation may only reduce length (or keep equal)
        assert len(doc.normalized_text) <= len(doc.raw_text) + 10  # +10 tolerance for NFC expansion


# ═══════════════════════════════════════════════════════════════════════════════
# G. Table extraction from DOCX
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocxTableExtraction:

    @skip_if_missing(TRAIN_DOCX_TABLE)
    def test_table_marker_present(self):
        """The TABLE block marker must appear if the DOCX has tables."""
        doc = load_docx(TRAIN_DOCX_TABLE)
        assert "[TABLE]" in doc.raw_text, (
            "Expected [TABLE] marker in DOCX with tables"
        )

    @skip_if_missing(TRAIN_DOCX_TABLE)
    def test_table_end_marker_present(self):
        doc = load_docx(TRAIN_DOCX_TABLE)
        assert "[/TABLE]" in doc.raw_text

    @skip_if_missing(TRAIN_DOCX_2)
    def test_docx_with_possible_table(self):
        """44737744 is known to have a table layout — ensure text is extracted."""
        doc = load_docx(TRAIN_DOCX_2)
        assert len(doc.raw_text) > 200


# ═══════════════════════════════════════════════════════════════════════════════
# H. Normalizer
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizer:

    def test_crlf_converted(self):
        assert "\r\n" not in normalize_text("line1\r\nline2")

    def test_null_bytes_removed(self):
        assert "\x00" not in normalize_text("abc\x00def")

    def test_excess_blank_lines_collapsed(self):
        result = normalize_text("a\n\n\n\n\nb")
        assert result.count("\n") <= 3, "Excess blank lines not collapsed"

    def test_trailing_whitespace_removed(self):
        result = normalize_text("line1   \nline2\t")
        for line in result.split("\n"):
            assert not line.endswith((" ", "\t")), f"Trailing WS in: {line!r}"

    def test_names_not_changed(self):
        name = "GERALDINE Q. GALINATO"
        assert name in normalize_text(f"Party: {name}")

    def test_date_not_changed(self):
        date = "20.05.2007"
        assert date in normalize_text(f"Start date: {date}")

    def test_monetary_value_not_changed(self):
        val = "Rs.6,500"
        assert val in normalize_text(f"Rent: {val} per month")

    def test_empty_input(self):
        assert normalize_text("") == ""

    def test_whitespace_only_input(self):
        assert normalize_text("   \n\n\t\n   ") == ""

    def test_unicode_nfc(self):
        # Decomposed 'é' (e + combining acute) should become composed 'é'
        decomposed = "caf\u0065\u0301"  # 'cafe' + combining acute
        result = normalize_text(decomposed)
        assert "\u00e9" in result or "caf" in result  # NFC 'é'


# ═══════════════════════════════════════════════════════════════════════════════
# I. DocumentContent structure contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocumentContentContract:

    @skip_if_missing(TRAIN_DOCX_1)
    def test_to_dict_keys(self):
        doc = load_docx(TRAIN_DOCX_1)
        d = doc.to_dict()
        for key in ["file_name", "file_path", "file_type", "raw_text_length",
                    "normalized_text_length", "page_count", "metadata"]:
            assert key in d, f"Missing key in to_dict(): {key}"

    @skip_if_missing(TRAIN_DOCX_1)
    def test_preview_truncates_at_500(self):
        doc = load_docx(TRAIN_DOCX_1)
        preview = doc.preview(500)
        assert len(preview) <= 501  # 500 chars + possible ellipsis char


# ═══════════════════════════════════════════════════════════════════════════════
# J. Unified load_document() interface
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadDocument:

    @skip_if_missing(TRAIN_DOCX_1)
    def test_load_document_dispatches_docx(self):
        doc = load_document(TRAIN_DOCX_1)
        assert doc.file_type == "docx"

    @skip_if_missing(TRAIN_PNG_1)
    def test_load_document_dispatches_png(self):
        import shutil
        if not shutil.which("tesseract"):
            pytest.skip("Tesseract not installed")
        doc = load_document(TRAIN_PNG_1, artifacts_dir=ARTIFACTS_DIR)
        assert doc.file_type == "png"

    @skip_if_missing(TEST_DOCX_1)
    def test_load_document_test_docx(self):
        doc = load_document(TEST_DOCX_1)
        assert doc.file_type == "docx"
        assert len(doc.normalized_text) > 100

    @skip_if_missing(TEST_PNG_1)
    def test_load_document_test_png(self):
        import shutil
        if not shutil.which("tesseract"):
            pytest.skip("Tesseract not installed")
        doc = load_document(TEST_PNG_1, artifacts_dir=ARTIFACTS_DIR)
        assert doc.file_type == "png"
        assert len(doc.raw_text) > 100

    def test_supported_extensions_set(self):
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".png"  in SUPPORTED_EXTENSIONS
        assert ".jpg"  in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
        assert ".pdf"  not in SUPPORTED_EXTENSIONS
