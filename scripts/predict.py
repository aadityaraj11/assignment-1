"""
scripts/predict.py
==================
Runs the LLM metadata extraction pipeline on a specific document.
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.pipeline import extract_metadata


def main():
    parser = argparse.ArgumentParser(description="Predict metadata for a document.")
    parser.add_argument("--file", required=True, help="Path to the document to process.")
    args = parser.parse_args()

    target_path = Path(args.file)
    if not target_path.exists():
        sys.exit(f"File not found: {target_path}")

    print("=" * 80)
    print(" PREDICTION PIPELINE")
    print("=" * 80)

    print(f"\nDOCUMENT : {target_path.resolve()}")

    artifacts_dir = PROJECT_DIR / "artifacts"

    # Run extraction
    prediction, examples, retrieval_scores, debug_info = extract_metadata(target_path, artifacts_dir)

    print(f"\nPROVIDER : {debug_info.get('provider')} (model: {debug_info.get('model')})")

    # ── 1. Retrieved examples + similarity scores ──────────────────────────────
    print("\n" + "─" * 80)
    print(" RETRIEVED EXAMPLES")
    print("─" * 80)
    if not retrieval_scores:
        print("  (none — index empty)")
    else:
        for i, (fname, score) in enumerate(retrieval_scores, 1):
            print(f"{i}. {fname} — similarity: {score:.4f}")

    # ── 2. Prompt preview (no secrets, truncated) ────────────────────
    print("\n" + "─" * 80)
    print(" PROMPT PREVIEW")
    print("─" * 80)
    prompt = debug_info.get("prompt", "")
    prompt_lines = prompt.replace("\r", "").strip()
    preview = prompt_lines[:1000]
    print(preview)
    if len(prompt_lines) > 1000:
        print(f"\n  ... [{len(prompt_lines) - 1000} chars truncated] ...")
        print("\n  [PROMPT TAIL]\n" + prompt_lines[-250:])

    # ── 3. Gemini structured JSON response ────────────────────────────────────
    print("\n" + "─" * 80)
    print(" PREDICTION")
    print("─" * 80)
    print(json.dumps(prediction, indent=2))

    # ── 4. Pydantic validation ─────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print(" VALIDATION")
    print("─" * 80)
    from app.schemas import MetadataExtraction
    try:
        validated = MetadataExtraction(**prediction)
        print(" PASS — all fields match MetadataExtraction schema.")
        none_fields = [k for k, v in validated.model_dump().items() if v is None]
        if none_fields:
            print(f"  Fields returned as null: {none_fields}")
    except Exception as e:
        print(f" FAIL — {e}")

    # ── 5. API call count ──────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print(" API CALLS MADE: 1")
    print("=" * 80)


if __name__ == "__main__":
    main()
