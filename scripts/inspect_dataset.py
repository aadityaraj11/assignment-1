"""
scripts/inspect_dataset.py
==========================
Dataset inspection script (Phase 1 + Phase 2).

Default mode  – Phase-1 CSV audit (fast, no OCR):
    python scripts/inspect_dataset.py

Phase-2 text extraction mode (runs loaders + OCR):
    python scripts/inspect_dataset.py --extract-text

Optional flags:
    --split train|test   Only inspect one split (default: both)
    --file <filename>    Inspect a single file by name
    --artifacts-dir PATH Where to write OCR artefacts (default: artifacts/)
"""

import argparse
import logging
import os
import sys
import textwrap
from pathlib import Path

# ── make project root importable ─────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# ── dependency guard ────────────────────────────────────────────────────────
try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not installed – run: pip install pandas")

try:
    from docx import Document as DocxDocument
except ImportError:
    sys.exit("python-docx not installed – run: pip install python-docx")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not installed – run: pip install pillow")

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = PROJECT_DIR / "data"
TRAIN_DIR   = DATA_DIR / "train"
TEST_DIR    = DATA_DIR / "test"
TRAIN_CSV   = DATA_DIR / "train.csv"
TEST_CSV    = DATA_DIR / "test.csv"
DEFAULT_ARTIFACTS = PROJECT_DIR / "artifacts"

SEPARATOR = "=" * 80
SEP2      = "-" * 80

# ── helpers ──────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def subsection(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


def wrap(text: str, width: int = 100, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)


def read_docx(path: Path) -> dict:
    """Return paragraphs, tables, and full raw text from a DOCX file."""
    doc = DocxDocument(str(path))

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text  = "\n".join(p.text for p in doc.paragraphs)

    tables = []
    for tbl in doc.tables:
        rows = []
        for row in tbl.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(cells)
        tables.append(rows)

    return {
        "paragraphs": paragraphs,
        "full_text": full_text,
        "tables": tables,
        "para_count": len(paragraphs),
        "table_count": len(tables),
        "char_count": len(full_text),
        "word_count": len(full_text.split()),
    }


def read_image_meta(path: Path) -> dict:
    """Return image metadata without loading full pixel data."""
    with Image.open(str(path)) as img:
        return {
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "file_size_kb": round(path.stat().st_size / 1024, 1),
        }


# ── CSV audit ────────────────────────────────────────────────────────────────

def audit_csv(csv_path: Path, label: str) -> pd.DataFrame:
    section(f"CSV AUDIT — {label}  [{csv_path.name}]")

    df = pd.read_csv(csv_path)
    # strip whitespace from all string columns
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

    print(f"\n  Rows       : {len(df)}")
    print(f"  Columns    : {list(df.columns)}")

    subsection("Column data types")
    print(df.dtypes.to_string())

    subsection("Missing values per column")
    na = df.isna().sum()
    print(na.to_string())
    if na.sum() == 0:
        print("  ✓ No missing values detected")

    subsection("Duplicate rows")
    dups = df.duplicated()
    print(f"  Fully duplicated rows: {dups.sum()}")

    subsection("Value distribution – Agreement Value")
    col = "Aggrement Value"
    if col in df.columns:
        print(df[col].describe().to_string())
        print(f"\n  Raw values: {df[col].tolist()}")

    subsection("Value distribution – Renewal Notice (Days)")
    col = "Renewal Notice (Days)"
    if col in df.columns:
        print(df[col].describe().to_string())
        print(f"\n  Raw values: {df[col].tolist()}")

    subsection("Date fields (raw string values)")
    for col in ["Aggrement Start Date", "Aggrement End Date"]:
        if col in df.columns:
            print(f"\n  {col}:")
            for v in df[col].tolist():
                print(f"    • {v!r}")

    subsection("Party fields (raw string values)")
    for col in ["Party One", "Party Two"]:
        if col in df.columns:
            print(f"\n  {col}:")
            for v in df[col].tolist():
                print(f"    • {v!r}")

    subsection("Full DataFrame (all rows)")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(df.to_string(index=False))

    # ── flag anomalies ──────────────────────────────────────────────────────
    subsection("Anomaly flags")
    anomalies = []

    # typo in column name
    if "Aggrement Value" in df.columns:
        anomalies.append("⚠ Column header typo: 'Aggrement' instead of 'Agreement' (both CSVs)")

    # impossible dates
    impossible_dates = [
        ("31.11.2009", "Nov has 30 days"),
        ("31.04.2011", "Apr has 30 days"),
        ("31.02.2011", "Feb never has 31 days"),
        ("31.03.2009", "OK – just note"),
    ]
    for col in ["Aggrement Start Date", "Aggrement End Date"]:
        if col in df.columns:
            for val in df[col].tolist():
                if isinstance(val, str):
                    for bad, reason in impossible_dates:
                        if bad in val:
                            anomalies.append(f"⚠ Impossible date '{bad}' in [{col}] — {reason}")

    # empty renewal notice
    col = "Renewal Notice (Days)"
    if col in df.columns and df[col].isna().any():
        anomalies.append(f"⚠ [{col}] has {df[col].isna().sum()} missing value(s)")

    # trailing whitespace (before strip)
    raw_df = pd.read_csv(csv_path)
    for c in raw_df.columns:
        if raw_df[c].dtype == object:
            has_trail = raw_df[c].dropna().apply(lambda v: v != v.strip()).any()
            if has_trail:
                anomalies.append(f"⚠ Column [{c}] has trailing/leading whitespace in raw CSV")

    if anomalies:
        for a in anomalies:
            print(f"  {a}")
    else:
        print("  ✓ No anomalies detected")

    return df


# ── DOCX audit ───────────────────────────────────────────────────────────────

def audit_docx_files(docx_files: list[Path], split: str, gt_df: pd.DataFrame) -> None:
    section(f"DOCX AUDIT — {split.upper()} split  ({len(docx_files)} files)")

    for fpath in sorted(docx_files):
        subsection(f"{fpath.name}  [{round(fpath.stat().st_size/1024,1)} KB]")

        try:
            info = read_docx(fpath)
        except Exception as e:
            print(f"  ERROR reading file: {e}")
            continue

        print(f"  Paragraphs : {info['para_count']}")
        print(f"  Tables     : {info['table_count']}")
        print(f"  Characters : {info['char_count']}")
        print(f"  Words      : {info['word_count']}")

        # ── paragraph preview ──
        print("\n  [First 8 non-empty paragraphs]")
        for i, p in enumerate(info["paragraphs"][:8]):
            print(wrap(repr(p)))

        if info["paragraphs"][8:16]:
            print("\n  [Paragraphs 9-16]")
            for p in info["paragraphs"][8:16]:
                print(wrap(repr(p)))

        # ── table content ──
        if info["tables"]:
            print(f"\n  [Tables — {info['table_count']} found]")
            for t_idx, tbl in enumerate(info["tables"]):
                print(f"    Table {t_idx + 1} ({len(tbl)} rows × {len(tbl[0]) if tbl else 0} cols):")
                for row in tbl[:6]:
                    print(f"      {row}")
                if len(tbl) > 6:
                    print(f"      … {len(tbl)-6} more rows")

        # ── ground-truth cross-reference ──
        stem = fpath.stem
        # match on filename stem (without extension)
        row = gt_df[gt_df["File Name"].str.strip() == stem]
        if not row.empty:
            row = row.iloc[0]
            print(f"\n  [Ground-truth values for this document]")
            print(f"    Agreement Value      : {row.get('Aggrement Value', 'N/A')}")
            print(f"    Start Date           : {row.get('Aggrement Start Date', 'N/A')}")
            print(f"    End Date             : {row.get('Aggrement End Date', 'N/A')}")
            print(f"    Renewal Notice (Days): {row.get('Renewal Notice (Days)', 'N/A')}")
            print(f"    Party One            : {row.get('Party One', 'N/A')}")
            print(f"    Party Two            : {row.get('Party Two', 'N/A')}")

            # surface value mentions inside document text (no regex extraction)
            full = info["full_text"].lower()
            val  = str(row.get("Aggrement Value", "")).strip()
            p1   = str(row.get("Party One", "")).strip().lower()
            p2   = str(row.get("Party Two", "")).strip().lower()

            print(f"\n  [Naive presence checks in document text]")
            print(f"    Value '{val}' found in text : {'YES' if val and val in full else 'NO (possible mismatch)'}")
            p1_found = any(tok in full for tok in p1.split() if len(tok) > 3)
            p2_found = any(tok in full for tok in p2.split() if len(tok) > 3)
            print(f"    Party One tokens present  : {'YES (partial)' if p1_found else 'NO'}")
            print(f"    Party Two tokens present  : {'YES (partial)' if p2_found else 'NO'}")
        else:
            print(f"\n  [No ground-truth row found for stem '{stem}' in {split}.csv]")

        # ── financial keyword surface scan ──
        full_lower = info["full_text"].lower()
        fin_terms = [
            "rent", "deposit", "advance rent", "security deposit",
            "monthly rent", "lease", "renewal", "termination",
        ]
        found_terms = [t for t in fin_terms if t in full_lower]
        print(f"\n  [Financial/legal keywords present]: {found_terms if found_terms else 'None of the tested terms'}")


# ── image audit ──────────────────────────────────────────────────────────────

def audit_image_files(img_files: list[Path], split: str, gt_df: pd.DataFrame) -> None:
    section(f"IMAGE AUDIT — {split.upper()} split  ({len(img_files)} files)")

    for fpath in sorted(img_files):
        subsection(f"{fpath.name}")
        try:
            meta = read_image_meta(fpath)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Format      : {meta['format']}")
        print(f"  Mode        : {meta['mode']}")
        print(f"  Dimensions  : {meta['width']} × {meta['height']} px")
        print(f"  File size   : {meta['file_size_kb']} KB")

        dpi_note = ""
        if meta["width"] > 2000 or meta["height"] > 2000:
            dpi_note = "high-res – OCR should be feasible"
        elif meta["width"] < 1000:
            dpi_note = "⚠ low-res – OCR quality risk"
        else:
            dpi_note = "medium-res"
        print(f"  Resolution  : {dpi_note}")

        stem = fpath.stem
        row  = gt_df[gt_df["File Name"].str.strip() == stem]
        if not row.empty:
            row = row.iloc[0]
            print(f"\n  [Ground-truth for this image]")
            print(f"    Agreement Value      : {row.get('Aggrement Value', 'N/A')}")
            print(f"    Start Date           : {row.get('Aggrement Start Date', 'N/A')}")
            print(f"    End Date             : {row.get('Aggrement End Date', 'N/A')}")
            print(f"    Renewal Notice (Days): {row.get('Renewal Notice (Days)', 'N/A')}")
            print(f"    Party One            : {row.get('Party One', 'N/A')}")
            print(f"    Party Two            : {row.get('Party Two', 'N/A')}")
        else:
            print(f"\n  [No ground-truth row found for stem '{stem}']")


# ── cross-split overlap check ────────────────────────────────────────────────

def check_cross_split_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    section("CROSS-SPLIT OVERLAP CHECK (train ↔ test)")

    train_stems = set(train_df["File Name"].str.strip().tolist())
    test_stems  = set(test_df["File Name"].str.strip().tolist())
    overlap     = train_stems & test_stems

    print(f"\n  Train file stems : {len(train_stems)}")
    print(f"  Test  file stems : {len(test_stems)}")
    print(f"  Overlap          : {len(overlap)}")
    if overlap:
        for s in overlap:
            print(f"    ⚠ '{s}' appears in BOTH train and test CSVs")
    else:
        print("  ✓ No overlap between train and test sets")

    # check if any test document appears to share content with train
    print("\n  [Checking for near-duplicate filenames across splits]")
    for t in test_stems:
        for r in train_stems:
            # simple heuristic: share ≥4 common tokens
            t_toks = set(t.lower().replace("-", " ").split())
            r_toks = set(r.lower().replace("-", " ").split())
            common = t_toks & r_toks - {"rental", "agreement", "pdf"}
            if len(common) >= 3:
                print(f"  ⚠ Possible content overlap: test='{t}' ↔ train='{r}' (shared tokens: {common})")


# ── summary statistics ───────────────────────────────────────────────────────

def print_summary(train_df, test_df, train_docx, train_imgs, test_docx, test_imgs):
    section("DATASET SUMMARY")

    total_train_docs = len(train_docx) + len(train_imgs)
    total_test_docs  = len(test_docx)  + len(test_imgs)

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │                    DATASET OVERVIEW                         │
  ├───────────────────────┬────────────┬────────────────────────┤
  │ Attribute             │  TRAIN     │  TEST                  │
  ├───────────────────────┼────────────┼────────────────────────┤
  │ CSV rows              │  {len(train_df):<10} │  {len(test_df):<22} │
  │ DOCX documents        │  {len(train_docx):<10} │  {len(test_docx):<22} │
  │ Image documents       │  {len(train_imgs):<10} │  {len(test_imgs):<22} │
  │ Total documents       │  {total_train_docs:<10} │  {total_test_docs:<22} │
  ├───────────────────────┴────────────┴────────────────────────┤
  │ Target fields (6):                                          │
  │   Agreement Value, Aggrement Start Date, Aggrement End Date │
  │   Renewal Notice (Days), Party One, Party Two              │
  ├─────────────────────────────────────────────────────────────┤
  │ NOTE: Column header uses 'Aggrement' (typo) not 'Agreement' │
  └─────────────────────────────────────────────────────────────┘
""")


# ── Phase-2 text extraction ───────────────────────────────────────────────────

def run_text_extraction(
    files: list[Path],
    artifacts_dir: Path,
    preview_chars: int = 500,
) -> None:
    """
    Run the Phase-2 ingestion pipeline on a list of document files and
    print a human-readable report: file type, text lengths, and previews.
    Saves OCR artefacts to artifacts_dir/ocr/<stem>.txt for images.
    """
    try:
        from app.loaders import load_document
    except ImportError as exc:
        print(f"  ERROR: Cannot import app.loaders — {exc}")
        print("  Ensure you are running from the project root.")
        return

    section(f"PHASE-2 TEXT EXTRACTION  ({len(files)} files)")

    for fpath in files:
        print(f"\n{SEP2}")
        print(f"  FILE : {fpath.name}")
        print(SEP2)

        try:
            doc = load_document(fpath, artifacts_dir=artifacts_dir)
        except Exception as exc:
            print(f"  ERROR loading '{fpath.name}': {exc}")
            continue

        print(f"  TYPE               : {doc.file_type.upper()}")
        print(f"  RAW TEXT LENGTH    : {len(doc.raw_text):,} chars")
        print(f"  NORMALIZED LENGTH  : {len(doc.normalized_text):,} chars")

        if doc.file_type in {"png", "jpg", "jpeg"}:
            print(f"  IMAGE DIMENSIONS   : {doc.metadata.get('image_width')}×{doc.metadata.get('image_height')} px")
            print(f"  OCR STRATEGY       : {doc.metadata.get('ocr_strategy', 'N/A')}")
            print(f"  OCR TEXT LENGTH    : {len(doc.raw_text):,} chars")
            print(f"\n  OCR PREVIEW (first {preview_chars} chars):")
            preview_text = (doc.raw_text[:preview_chars] + "…") if len(doc.raw_text) > preview_chars else doc.raw_text
        else:
            print(f"  PARAGRAPH COUNT    : {doc.metadata.get('paragraph_count', 'N/A')}")
            print(f"  PAGE COUNT         : {doc.page_count}")
            print(f"\n  PREVIEW (first {preview_chars} chars of normalized text):")
            preview_text = doc.preview(preview_chars)

        # Indent the preview for readability
        for line in preview_text.splitlines():
            print(f"    {line}")

        if doc.file_type in {"png", "jpg", "jpeg"}:
            ocr_path = artifacts_dir / "ocr" / f"{fpath.stem}.txt"
            if ocr_path.exists():
                print(f"\n  OCR artefact saved → {ocr_path.relative_to(PROJECT_DIR)}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset inspection + optional text extraction."
    )
    parser.add_argument(
        "--extract-text",
        action="store_true",
        help="Run Phase-2 loaders; print text lengths and previews for all documents.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which data split to inspect (default: both).",
    )
    parser.add_argument(
        "--file",
        metavar="FILENAME",
        default=None,
        help="Inspect a single file by name (e.g. 18325926-Rental-Agreement-1.docx).",
    )
    parser.add_argument(
        "--artifacts-dir",
        metavar="PATH",
        default=str(DEFAULT_ARTIFACTS),
        help=f"Directory for OCR artefacts (default: {DEFAULT_ARTIFACTS}).",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Number of preview characters to print per document (default: 500).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging from loaders.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    artifacts_dir = Path(args.artifacts_dir)

    print(SEPARATOR)
    print("  META-DATA EXTRACTION — DATASET INSPECTION")
    print(f"  Data directory    : {DATA_DIR}")
    print(f"  Artefacts dir     : {artifacts_dir}")
    print(f"  Mode              : {'CSV audit + text extraction' if args.extract_text else 'CSV audit only'}")
    print(SEPARATOR)

    # Validate paths
    for p in [TRAIN_CSV, TEST_CSV, TRAIN_DIR, TEST_DIR]:
        if not p.exists():
            sys.exit(f"ERROR: Expected path does not exist: {p}")

    # ── Load CSVs ────────────────────────────────────────────────────────────
    train_df = audit_csv(TRAIN_CSV, "TRAIN")
    test_df  = audit_csv(TEST_CSV,  "TEST")

    # ── Categorise documents ─────────────────────────────────────────────────
    train_docx = sorted(TRAIN_DIR.glob("*.docx"))
    train_imgs = sorted(TRAIN_DIR.glob("*.png")) + sorted(TRAIN_DIR.glob("*.jpg"))
    test_docx  = sorted(TEST_DIR.glob("*.docx"))
    test_imgs  = sorted(TEST_DIR.glob("*.png")) + sorted(TEST_DIR.glob("*.jpg"))

    # ── Audit documents ──────────────────────────────────────────────────────
    combined_gt = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset=["File Name"])

    if args.split in ("both", "train"):
        audit_docx_files(train_docx, "train", combined_gt)
        audit_image_files(train_imgs, "train", combined_gt)
    if args.split in ("both", "test"):
        audit_docx_files(test_docx,  "test",  combined_gt)
        audit_image_files(test_imgs,  "test",  combined_gt)

    # ── Cross-split overlap ─────────────────────────────────────────────────
    check_cross_split_overlap(train_df, test_df)

    # ── Summary ─────────────────────────────────────────────────────────────
    print_summary(train_df, test_df, train_docx, train_imgs, test_docx, test_imgs)

    section("KEY DATASET PROBLEMS (Phase 1 findings)")
    print("""
  1.  'Aggrement' typo in column headers (both CSVs)
  2.  Impossible end dates: 31.11.2009, 31.02.2011, 31.04.2011
  3.  Renewal Notice missing for row 44737744
  4.  24158401 appears in BOTH train and test (data leakage)
  5.  Pervasive leading/trailing whitespace in Party One/Two cells
  6.  Advance/deposit amounts in documents may differ from monthly rent in CSV
  7.  Trailing space in start date for 50070534: '01.04.2010 '
  8.  Only 10 training examples → zero/few-shot LLM required
  9.  2 test documents are PNG scans → OCR mandatory
  10. 6683127 is a 1-page subset of 6683129 (same agreement)
""")

    # ── Phase-2 text extraction ───────────────────────────────────────────────
    if args.extract_text:
        if args.file:
            # Single-file mode
            candidates = (
                list(TRAIN_DIR.glob(args.file)) +
                list(TEST_DIR.glob(args.file))
            )
            if not candidates:
                print(f"  File '{args.file}' not found in train/ or test/.")
            else:
                run_text_extraction(candidates, artifacts_dir, args.preview_chars)
        else:
            # All files in selected split(s)
            all_files: list[Path] = []
            if args.split in ("both", "train"):
                all_files += train_docx + train_imgs
            if args.split in ("both", "test"):
                all_files += test_docx + test_imgs
            run_text_extraction(all_files, artifacts_dir, args.preview_chars)
    else:
        print("\n  Tip: run with --extract-text to also run the Phase-2 loader pipeline.")

    print(f"\n{SEPARATOR}")
    print("  Inspection complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
