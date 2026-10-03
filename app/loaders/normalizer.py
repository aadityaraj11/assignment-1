"""
app/loaders/normalizer.py
=========================
Conservative text normalisation.

Rules applied (in order):
  1. Replace Windows/old-Mac line endings with Unix newlines.
  2. Remove null bytes and other non-printable control characters
     (except tab and newline which are meaningful layout characters).
  3. Collapse runs of 3+ consecutive blank lines down to 2 blank lines
     (preserve section breaks without excessive whitespace).
  4. Strip trailing whitespace per line (keeps leading indent if present).
  5. Normalise Unicode: decompose then recompose (NFC) for safe string
     comparison downstream.
  6. Strip leading/trailing whitespace from the whole document.

Rules deliberately NOT applied:
  - No word substitution or spelling correction.
  - No removal of short/single tokens that might be meaningful.
  - No flattening of all newlines — layout carries semantic information.
  - No lowercasing — casing is significant for name matching.
  - No removal of punctuation — dates and monetary values depend on it.
"""

from __future__ import annotations

import re
import unicodedata


# Control characters other than TAB (\x09) and LF (\x0A)
_CTRL_CHAR = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

# 3 or more consecutive blank lines → 2 blank lines
_EXCESS_BLANK = re.compile(r"(\n\s*){3,}", re.MULTILINE)

# Trailing whitespace on each line (keep leading to preserve indentation)
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)

# Multiple spaces within a line (NOT at line start) → single space
# This catches runs of spaces like "name:     value" but NOT indentation.
_INLINE_SPACES = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")


def normalize_text(text: str) -> str:
    """
    Apply conservative normalisation to document text.

    Parameters
    ----------
    text : str
        Raw text as extracted by a loader.

    Returns
    -------
    str
        Normalised text safe for downstream processing.
    """
    if not text:
        return ""

    # 1. Unify line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove control characters (preserve \t and \n)
    text = _CTRL_CHAR.sub("", text)

    # 3. Normalise Unicode to NFC (safe canonical composition)
    text = unicodedata.normalize("NFC", text)

    # 4. Strip trailing whitespace per line
    text = _TRAILING_WS.sub("", text)

    # 5. Collapse excess blank lines
    text = _EXCESS_BLANK.sub("\n\n", text)

    # 6. Collapse mid-line multi-spaces (does not touch line starts)
    text = _INLINE_SPACES.sub(" ", text)

    # 7. Final strip
    return text.strip()
