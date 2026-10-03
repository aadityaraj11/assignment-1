"""
app/loaders/__init__.py
=======================
Public interface for the document ingestion layer.

Usage
-----
    from app.loaders import load_document

    doc = load_document("data/train/example.docx")
    print(doc.normalized_text[:200])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.loaders.base import DocumentContent  # noqa: F401  (re-exported)
from app.loaders.docx_loader import load_docx
from app.loaders.image_loader import SUPPORTED_IMAGE_EXTENSIONS, load_image

# All supported extensions mapped to their loader
_DOCX_EXTENSIONS  = {".docx"}
_IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS

SUPPORTED_EXTENSIONS = _DOCX_EXTENSIONS | _IMAGE_EXTENSIONS


def load_document(
    path: str | Path,
    *,
    artifacts_dir: Optional[str | Path] = None,
) -> DocumentContent:
    """
    Load ANY supported document and return a DocumentContent object.

    File type is determined purely by file extension — NOT by filename content.

    Parameters
    ----------
    path : str | Path
        Path to the document file.
    artifacts_dir : str | Path | None
        If provided, image OCR artefacts are saved here for debugging.
        Ignored for DOCX files.

    Returns
    -------
    DocumentContent

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in _DOCX_EXTENSIONS:
        return load_docx(path)

    if suffix in _IMAGE_EXTENSIONS:
        return load_image(
            path,
            artifacts_dir=Path(artifacts_dir) if artifacts_dir else None,
        )

    raise ValueError(
        f"Unsupported file extension '{suffix}' for '{path.name}'. "
        f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
    )


__all__ = ["load_document", "DocumentContent", "SUPPORTED_EXTENSIONS"]
