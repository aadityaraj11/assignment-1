"""
app/loaders/image_loader.py
===========================
Image ingestion layer: loads PNG / JPG / JPEG files, applies adaptive
preprocessing, runs Tesseract OCR, and returns a DocumentContent object.

Design principles
-----------------
* Works for arbitrary scanned documents — no hardcoded assumptions about
  content, language, or layout specific to this dataset.
* Conservative preprocessing: tries a sequence of strategies and picks
  the one that yields the most text, rather than applying a single fixed
  pipeline that may degrade quality on diverse inputs.
* Preserves line breaks because layout is semantically meaningful for
  later extraction (party names on their own line, dates following
  keywords, etc.).
* No field extraction, no regex parsing, no business logic.

Dependencies
------------
  pip install pytesseract opencv-python-headless Pillow
  sudo apt install tesseract-ocr tesseract-ocr-eng
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from app.loaders.normalizer import normalize_text

if TYPE_CHECKING:
    from app.loaders.base import DocumentContent

log = logging.getLogger(__name__)

# Tesseract configuration: page-segmentation mode 3 (fully automatic layout
# analysis), OEM 3 (LSTM + legacy; falls back gracefully).
_TESS_CONFIG = r"--oem 3 --psm 3"

# Supported image extensions (lower-case).
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing strategies
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR OpenCV image to grayscale."""
    if len(img_bgr.shape) == 2:
        return img_bgr  # already grayscale
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """
    Apply a mild Non-Local Means denoising pass.
    h=10 is conservative — it reduces noise without blurring ink strokes.
    """
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def _scale_to_300dpi(img_bgr: np.ndarray, current_dpi: Optional[int] = None) -> np.ndarray:
    """
    If the image is smaller than 300 DPI equivalent (heuristic: height < 1600px
    for an A4 page), upscale with INTER_CUBIC to improve OCR accuracy.
    We do NOT downscale because high-res images already work well with Tesseract.
    """
    h, w = img_bgr.shape[:2]
    # Heuristic: if portrait A4 at 300dpi ≈ 2480×3508 px.
    # Only upscale if the image looks significantly smaller.
    if h < 1500 and w < 1200:
        scale = max(1600 / h, 1200 / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        log.debug("Upscaling %dx%d → %dx%d for better OCR", w, h, new_w, new_h)
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img_bgr


def _strategy_direct(gray: np.ndarray) -> np.ndarray:
    """Strategy 0: grayscale only — best for already-clean scans."""
    return gray


def _strategy_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Strategy 1: CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Improves contrast locally without over-brightening uniform regions.
    Works well for lightly faded or unevenly lit scans.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _strategy_adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Strategy 2: Gaussian adaptive thresholding.
    Produces a binary image; effective for documents with uneven illumination.
    Block size 31 and C=10 are moderate defaults — not tuned for any specific doc.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def _strategy_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Strategy 3: Otsu global thresholding + Gaussian blur pre-smooth.
    Effective when the background is relatively uniform.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# Ordered list: (name, preprocessing function)
_STRATEGIES = [
    ("direct_gray",          _strategy_direct),
    ("clahe",                _strategy_clahe),
    ("adaptive_threshold",   _strategy_adaptive_threshold),
    ("otsu",                 _strategy_otsu),
]


# ─────────────────────────────────────────────────────────────────────────────
# OCR helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_tesseract(preprocessed: np.ndarray) -> str:
    """
    Run Tesseract on a preprocessed (gray or binary) numpy array.
    Returns raw OCR string.
    """
    # Convert numpy array → PIL Image (Tesseract wrapper expects PIL)
    pil_img = Image.fromarray(preprocessed)
    text: str = pytesseract.image_to_string(pil_img, config=_TESS_CONFIG, lang="eng")
    return text


def _text_quality_score(text: str) -> float:
    """
    Heuristic quality score for an OCR result.

    Higher is better.  We count printable non-whitespace characters.
    A secondary penalty is applied for a very high proportion of
    non-alphanumeric characters (OCR garbage).
    """
    if not text:
        return 0.0
    non_ws = sum(1 for c in text if not c.isspace())
    if non_ws == 0:
        return 0.0
    alnum = sum(1 for c in text if c.isalnum())
    alpha_ratio = alnum / non_ws
    # Weight: raw length × alpha quality
    return non_ws * (0.5 + 0.5 * alpha_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers (importable for testing)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Try all preprocessing strategies, run OCR on each, and return the
    (preprocessed_array, strategy_name) that produced the best text quality.

    Parameters
    ----------
    img_bgr : np.ndarray
        Image loaded with cv2.imread (BGR colour or grayscale).

    Returns
    -------
    tuple[np.ndarray, str]
        The best preprocessed array and the name of the winning strategy.
    """
    gray = _to_gray(img_bgr)
    gray = _denoise(gray)

    best_score = -1.0
    best_arr: np.ndarray = gray
    best_name: str = "direct_gray"

    for name, fn in _STRATEGIES:
        try:
            processed = fn(gray)
            text = _run_tesseract(processed)
            score = _text_quality_score(text)
            log.debug("Strategy %-22s → score %.1f (%d chars)", name, score, len(text))
            if score > best_score:
                best_score = score
                best_arr = processed
                best_name = name
        except Exception as exc:  # noqa: BLE001
            log.warning("Strategy %s failed: %s", name, exc)

    log.info("Best preprocessing strategy for image: %s (score %.1f)", best_name, best_score)
    return best_arr, best_name


def extract_ocr_text(img_bgr: np.ndarray) -> tuple[str, str]:
    """
    Full OCR pipeline for a single image.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image array (from cv2.imread).

    Returns
    -------
    tuple[str, str]
        (raw_ocr_text, winning_strategy_name)
    """
    img_bgr = _scale_to_300dpi(img_bgr)
    best_arr, strategy = preprocess_image(img_bgr)
    raw_text = _run_tesseract(best_arr)
    return raw_text, strategy


# ─────────────────────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path, artifacts_dir: Optional[Path] = None) -> "DocumentContent":
    """
    Load a PNG / JPG / JPEG image, run OCR, and return a DocumentContent.

    Parameters
    ----------
    path : Path
        Filesystem path to the image file.
    artifacts_dir : Optional[Path]
        If provided, save the raw OCR text to
        ``<artifacts_dir>/ocr/<stem>.txt`` for debugging.
        The original file is NOT modified.

    Returns
    -------
    DocumentContent

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the file cannot be read as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{suffix}'. "
            f"Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    # Load via OpenCV (handles PNG, JPG, BMP, TIFF)
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"cv2.imread could not read '{path.name}'. File may be corrupt.")

    h, w = img_bgr.shape[:2]
    log.info("Loaded image %s: %dx%d px", path.name, w, h)

    # Run OCR
    raw_text, strategy = extract_ocr_text(img_bgr)
    norm_text = normalize_text(raw_text)

    log.info(
        "OCR complete for %s — strategy=%s raw=%d chars normalized=%d chars",
        path.name, strategy, len(raw_text), len(norm_text),
    )

    # Optionally persist OCR output for debugging
    if artifacts_dir is not None:
        ocr_dir = Path(artifacts_dir) / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        out_path = ocr_dir / f"{path.stem}.txt"
        out_path.write_text(raw_text, encoding="utf-8")
        log.debug("OCR artifact saved → %s", out_path)

    file_type = suffix.lstrip(".")  # "png", "jpg", "jpeg"

    from app.loaders.base import DocumentContent
    return DocumentContent(
        file_name=path.name,
        file_path=str(path.resolve()),
        file_type=file_type,
        raw_text=raw_text,
        normalized_text=norm_text,
        page_count=1,
        metadata={
            "image_width": w,
            "image_height": h,
            "ocr_strategy": strategy,
            "raw_char_count": len(raw_text),
        },
    )
