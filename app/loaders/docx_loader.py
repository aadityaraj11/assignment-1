"""
app/loaders/docx_loader.py
==========================
DOCX ingestion layer.

Extracts:
  - paragraphs (headings + body text) in document order
  - tables: each cell linearised row-by-row with a TAB separator and rows
    separated by newlines, wrapped in a clear TABLE … END TABLE block so
    downstream readers can distinguish tabular content from prose.

Returns a DocumentContent dataclass.  No metadata extraction is performed here.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.loaders.normalizer import normalize_text

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _para_text(para: Paragraph) -> str:
    """Return the full text of a paragraph, preserving internal spaces."""
    return para.text  # python-docx joins runs automatically


def _table_to_text(table: Table) -> str:
    """
    Linearise a DOCX table into a human-readable text block.

    Format:
        [TABLE]
        cell00\tcell01\t...
        cell10\tcell11\t...
        [/TABLE]

    Merges duplicate adjacent cells caused by merged rows/columns in DOCX XML.
    """
    lines: list[str] = ["[TABLE]"]
    seen_rows: set[tuple[str, ...]] = set()

    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        # Deduplicate merged-cell artefacts (python-docx repeats the same
        # Cell object for horizontally spanned cells).
        deduped: list[str] = []
        prev = object()  # sentinel
        for c in cells:
            if c is not prev:  # identity check won't work on str – use value
                deduped.append(c)
            prev = c

        row_tuple = tuple(deduped)
        if row_tuple in seen_rows:
            continue  # skip vertically repeated rows from merged cells
        seen_rows.add(row_tuple)
        lines.append("\t".join(deduped))

    lines.append("[/TABLE]")
    return "\n".join(lines)


def _iter_block_items(doc: Document):
    """
    Yield top-level block items (Paragraph or Table) in document body order.

    python-docx's doc.paragraphs and doc.tables flatten the document and lose
    the interleaved order.  We walk the XML directly to preserve reading order.
    """
    body = doc.element.body
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            yield Paragraph(child, doc)
        elif tag == "tbl":
            yield Table(child, doc)
        # sdt (structured document tags), sectPr, etc. are ignored


# ─────────────────────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────────────────────

def load_docx(path: Path) -> "DocumentContent":
    """
    Load a DOCX file and return a DocumentContent object.

    Parameters
    ----------
    path : Path
        Filesystem path to the .docx file.

    Returns
    -------
    DocumentContent
        Contains raw_text (joined paragraphs + linearised tables in order),
        normalized_text, and basic metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed as a DOCX.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX not found: {path}")

    try:
        doc = Document(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot parse '{path.name}' as DOCX: {exc}") from exc

    parts: list[str] = []

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            txt = _para_text(block)
            if txt.strip():          # skip blank paragraphs
                parts.append(txt)
        elif isinstance(block, Table):
            parts.append(_table_to_text(block))

    raw_text = "\n".join(parts)
    norm_text = normalize_text(raw_text)

    log.debug(
        "load_docx %s → %d chars raw / %d chars normalised",
        path.name, len(raw_text), len(norm_text),
    )

    # Import here to avoid circular import at module load time
    from app.loaders.base import DocumentContent
    return DocumentContent(
        file_name=path.name,
        file_path=str(path.resolve()),
        file_type="docx",
        raw_text=raw_text,
        normalized_text=norm_text,
        page_count=None,    # reliable pagination not possible without rendering
        metadata={"paragraph_count": len(parts)},
    )
