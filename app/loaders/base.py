"""
app/loaders/base.py
===================
Shared data structures for the ingestion layer.

DocumentContent is the single output type produced by every loader.
Nothing in this module performs any field extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentContent:
    """
    Normalised representation of an ingested document.

    Attributes
    ----------
    file_name : str
        Basename of the source file (e.g. "example.docx").
    file_path : str
        Absolute filesystem path to the source file.
    file_type : str
        One of: "docx", "png", "jpg", "jpeg".
    raw_text : str
        Text exactly as extracted by the loader (before normalisation).
        For images this is the raw OCR output.
    normalized_text : str
        Lightly cleaned version of raw_text (see normalizer.py).
    page_count : Optional[int]
        Number of pages, if determinable (1 for images; None for DOCX).
    metadata : dict[str, Any]
        Loader-level metadata (e.g. image dimensions, paragraph count).
        Must NOT contain extracted business fields (dates, names, etc.).
    """

    file_name: str
    file_path: str
    file_type: str
    raw_text: str
    normalized_text: str
    page_count: Optional[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── convenience ──────────────────────────────────────────────────────────

    def preview(self, chars: int = 500) -> str:
        """Return the first `chars` characters of normalized_text."""
        text = self.normalized_text or self.raw_text
        return text[:chars] + ("…" if len(text) > chars else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "raw_text_length": len(self.raw_text),
            "normalized_text_length": len(self.normalized_text),
            "page_count": self.page_count,
            "metadata": self.metadata,
        }
