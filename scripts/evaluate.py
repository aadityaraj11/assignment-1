"""
scripts/evaluate.py
===================
Evaluation script: computes exact-match recall per field.
TODO: Implement in Phase 3 (after predictions are available).

Usage:
    python scripts/evaluate.py --predictions predictions.csv --ground_truth data/test.csv
"""
import argparse
import pandas as pd
from pathlib import Path


FIELDS = [
    "Aggrement Value",
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]


def normalise(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def evaluate(pred_csv: Path, gt_csv: Path) -> None:
    pred_df = pd.read_csv(pred_csv)
    gt_df   = pd.read_csv(gt_csv)

    merged = pred_df.merge(gt_df, on="File Name", suffixes=("_pred", "_gt"))

    results = {}
    for field in FIELDS:
        pred_col = f"{field}_pred"
        gt_col   = f"{field}_gt"
        if pred_col not in merged.columns or gt_col not in merged.columns:
            print(f"  ⚠ Field '{field}' missing in one of the CSVs — skipping")
            continue
        exact = (merged[pred_col].apply(normalise) == merged[gt_col].apply(normalise)).sum()
        total = len(merged)
        results[field] = exact / total if total else 0.0

    print("\nExact-match recall per field:")
    for field, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {field:<28} {score:.2%}  {bar}")
    overall = sum(results.values()) / len(results) if results else 0.0
    print(f"\n  Overall (macro-avg): {overall:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",   required=True)
    parser.add_argument("--ground_truth",  required=True)
    args = parser.parse_args()
    evaluate(Path(args.predictions), Path(args.ground_truth))
