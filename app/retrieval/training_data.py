"""
app/retrieval/training_data.py
==============================
Dataset adapter for loading training metadata & locating training files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    file_name: str
    document_text: str
    metadata: dict[str, Optional[str]] = field(default_factory=dict)


def _clean_val(val) -> Optional[str]:
    """Strip whitespace, handle pd.isna -> None"""
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def load_training_metadata(train_csv_path: str | Path) -> dict[str, dict[str, Optional[str]]]:
    """
    Load data/train.csv and return a mapping of:
    file_name -> metadata dict

    The dict keys will match the MetadataExtraction schema exactly.
    Handles the "Aggrement" typo safely.
    """
    path = Path(train_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    # Read CSV
    df = pd.read_csv(path)

    # Normalize column names internally
    col_map = {
        "Aggrement Value": "agreement_value",
        "Aggrement Start Date": "agreement_start_date",
        "Aggrement End Date": "agreement_end_date",
        "Renewal Notice (Days)": "renewal_notice_days",
        "Party One": "party_one",
        "Party Two": "party_two",
    }

    result = {}
    for _, row in df.iterrows():
        fname = _clean_val(row.get("File Name"))
        if not fname:
            continue

        meta = {}
        for csv_col, sys_col in col_map.items():
            if csv_col in df.columns:
                
                raw_val = row[csv_col]
                # If numeric like 60.0 for notice days, format as int first then str to drop '.0'
                if pd.notna(raw_val) and csv_col == "Renewal Notice (Days)" and isinstance(raw_val, float):
                    raw_val = str(int(raw_val))
                meta[sys_col] = _clean_val(raw_val)
            else:
                meta[sys_col] = None

        result[fname] = meta

    log.info("Loaded metadata for %d training examples", len(result))
    return result


def find_training_file(train_dir: str | Path, stem: str) -> Optional[Path]:
    """
    Locates the actual file corresponding to the 'File Name' stem 
    inside the training dir.
    """
    path = Path(train_dir)
    # The stem in CSV is without extension.
    # Supported extensions
    exts = [".docx", ".png", ".jpg", ".jpeg"]
    for ext in exts:
        target = path / f"{stem}{ext}"
        if target.exists():
            return target
            
    # Some test CSVs have ".pdf" in the stem but the file is ".pdf.docx"
    # Example: 156155545-Rental-Agreement-Kns-Home.pdf.docx is test, but just robust check:
    target_pdf_docx = path / f"{stem}.pdf.docx"
    if target_pdf_docx.exists():
        return target_pdf_docx
        
    return None
