"""
scripts/loocv.py
================
Leave-One-Out Cross-Validation over the 10 training examples.
Calculates zero/few-shot accuracy strictly withholding the target document.
Uses Gemini REST provider.
"""

import json
import sys
import math
import time
from pathlib import Path
import numpy as np

# Add project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.loaders import load_document
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.retriever import RetrieverIndex
from app.retrieval.training_data import load_training_metadata, find_training_file
from app.schemas import MetadataExtraction
from app.llm.provider import get_provider, GeminiRESTProvider
from app.llm.extractor import build_extraction_prompt

DATA_DIR = PROJECT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TRAIN_CSV = DATA_DIR / "train.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


def _normalize_val(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, str):
        v = val.strip()
        if not v or v.lower() == "nan":
            return None
        return v
    return val


def categorize_error(field: str, expected: str, predicted: str, doc_text: str) -> str:
    """Assigns a descriptive failure category based on document context and error type."""
    if predicted is None:
        if field == "renewal_notice_days":
            return "notice clause ambiguity"
        return "model extraction error"
    
    if field in ["agreement_start_date", "agreement_end_date"]:
        return "execution date vs commencement date" if "execution" in doc_text.lower() else "multiple dates"
    
    if field == "agreement_value":
        return "multiple monetary values"
        
    if field in ["party_one", "party_two"]:
        return "party identification ambiguity"
        
    return "model extraction error"


def main():
    provider = get_provider()
    model_name = getattr(provider, "model", "gemini-2.5-flash")
    top_k = 3
    
    print("=" * 80)
    print(" REAL GEMINI LOOCV EXPERIMENT")
    print("=" * 80)
    print(f"Provider: Gemini REST")
    print(f"Model: {model_name}")
    print(f"Number of validation folds: 10")
    print(f"Retrieval k: {top_k}")
    print("=" * 80)
    
    metadata_map = load_training_metadata(TRAIN_CSV)
    stems = list(metadata_map.keys())
    
    print(f"Loaded {len(stems)} training examples for LOOCV.\n")
    
    print("Pre-computing documents and embeddings ...")
    embedder = EmbeddingModel()
    
    doc_store = {}
    for stem in stems:
        print(f"Processing '{stem}'...", flush=True)
        fpath = find_training_file(TRAIN_DIR, stem)
        if not fpath:
            print(f" [!] Missing file for '{stem}'")
            continue
            
        doc = load_document(fpath)
        text = doc.normalized_text
        if not text.strip():
            print(f" [!] Empty text for {fpath.name}")
            continue
            
        emb = embedder.encode(text)
        
        # Clean ground truth values
        gt_cleaned = {k: _normalize_val(v) for k, v in metadata_map[stem].items()}
        
        doc_store[stem] = {
            "text": text,
            "emb": emb,
            "target": gt_cleaned,
            "fpath": fpath
        }
        
    valid_stems = list(doc_store.keys())
    print(f"\nSuccessfully processed {len(valid_stems)} documents for evaluation.")
    
    field_keys = [
        "agreement_value", "agreement_start_date", "agreement_end_date",
        "renewal_notice_days", "party_one", "party_two"
    ]
    
    hits_per_field = {k: 0 for k in field_keys}
    total_per_field = {k: 0 for k in field_keys}
    
    loocv_results = []
    
    total_api_calls = 0
    successful_calls = 0
    failed_calls = 0
    invalid_json_responses = 0
    
    for fold_idx, holdout_stem in enumerate(valid_stems, 1):
        print(f"\n[Fold {fold_idx}/10] Holding out: {holdout_stem}")
        
        # 1. Build index strictly excluding the holdout document
        train_stems = [s for s in valid_stems if s != holdout_stem]
        
        train_embeddings = []
        train_metadata = []
        for ts in train_stems:
            train_embeddings.append(doc_store[ts]["emb"])
            train_metadata.append({
                "file_name": ts,
                "document_text": doc_store[ts]["text"],
                "metadata": doc_store[ts]["target"]
            })
            
        temp_index = RetrieverIndex(ARTIFACTS_DIR / "temp_loocv_index")
        emb_matrix = np.vstack(train_embeddings)
        temp_index.save(emb_matrix, train_metadata)
        
        # 2. Retrieve top-k examples for holdout document
        query_emb = doc_store[holdout_stem]["emb"]
        search_results = temp_index.search(query_emb, top_k=top_k)
        
        retrieved_examples = [ex for ex, score in search_results]
        retrieved_file_names = [ex.file_name for ex in retrieved_examples]
        retrieved_with_scores = [
            {"file_name": ex.file_name, "similarity": round(float(score), 4)}
            for ex, score in search_results
        ]
        
        # 3. SELF-RETRIEVAL VERIFICATION ASSERTION
        assert holdout_stem not in retrieved_file_names, (
            f"CRITICAL ERROR: Self-retrieval detected in fold for {holdout_stem}! "
            f"Holdout file appeared in retrieved examples: {retrieved_file_names}"
        )
        ex_strs = [f"{r['file_name']} ({r['similarity']:.4f})" for r in retrieved_with_scores]
        print(f"  Retrieved examples: {', '.join(ex_strs)}")
        
        # 4. Build prompt and invoke Gemini REST provider
        target_text = doc_store[holdout_stem]["text"]
        prompt = build_extraction_prompt(target_text, retrieved_examples)
        
        # Pacing rate limit to stay under free tier 5 RPM
        if fold_idx > 1:
            print("  Pacing 13s to respect free-tier rate limits...", flush=True)
            time.sleep(13.0)
            
        total_api_calls += 1
        raw_dict = {}
        try:
            raw_dict = provider.generate_structured(prompt, MetadataExtraction)
            successful_calls += 1
        except ValueError as e:
            failed_calls += 1
            invalid_json_responses += 1
            print(f"  [!] Gemini returned invalid JSON or parsing failed: {e}")
        except Exception as e:
            failed_calls += 1
            print(f"  [!] Gemini REST API call failed: {e}")
            
        # Parse & clean predicted values
        prediction_obj = MetadataExtraction(**raw_dict) if raw_dict else MetadataExtraction()
        cleaned_pred = {}
        for k, v in prediction_obj.model_dump().items():
            cleaned_pred[k] = _normalize_val(v)
            
        actual = doc_store[holdout_stem]["target"]
        
        # 5. Exact match comparison per field
        doc_matches = {}
        for k in field_keys:
            pred_v = cleaned_pred.get(k)
            act_v = actual.get(k)
            
            total_per_field[k] += 1
            if pred_v == act_v:
                hits_per_field[k] += 1
                doc_matches[k] = True
            else:
                doc_matches[k] = False
                
        loocv_results.append({
            "file_name": holdout_stem,
            "retrieved_examples": retrieved_with_scores,
            "ground_truth": actual,
            "prediction": cleaned_pred,
            "field_matches": doc_matches,
            "document_text": target_text
        })
        
    print("\n" + "=" * 80)
    print(" REAL GEMINI LOOCV RESULTS TABLE")
    print("=" * 80)
    print(f"{'FIELD':<25} {'CORRECT':>10} {'TOTAL':>10} {'RECALL':>10}")
    print("-" * 80)
    
    macro_recalls = []
    for k in field_keys:
        correct = hits_per_field[k]
        total = total_per_field[k]
        recall = (correct / total) * 100 if total > 0 else 0.0
        macro_recalls.append(recall)
        print(f"{k:<25} {correct:>10} {total:>10} {recall:>9.1f}%")
        
    macro_avg_recall = float(np.mean(macro_recalls)) if macro_recalls else 0.0
    print("-" * 80)
    print(f"{'Macro-average recall':<25} {'':>10} {'':>10} {macro_avg_recall:>9.1f}%")
    print("=" * 80)
    
    # Save artifacts
    out_dir = ARTIFACTS_DIR / "loocv"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. predictions.json
    predictions_only = {r["file_name"]: r["prediction"] for r in loocv_results}
    with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions_only, f, indent=2)
        
    # 2. metrics.json
    metrics_summary = {
        "provider": "Gemini REST",
        "model": model_name,
        "total_api_calls": total_api_calls,
        "successful_calls": successful_calls,
        "failed_calls": failed_calls,
        "invalid_json_responses": invalid_json_responses,
        "macro_average_recall": round(macro_avg_recall, 2),
        "fields": {
            k: {
                "correct": hits_per_field[k],
                "total": total_per_field[k],
                "recall_percent": round((hits_per_field[k] / total_per_field[k] * 100), 2) if total_per_field[k] > 0 else 0.0
            }
            for k in field_keys
        }
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
        
    # 3. detailed_results.json (without text to keep clean)
    detailed_clean = [
        {
            "file_name": r["file_name"],
            "retrieved_examples": r["retrieved_examples"],
            "ground_truth": r["ground_truth"],
            "prediction": r["prediction"],
            "field_matches": r["field_matches"]
        }
        for r in loocv_results
    ]
    with open(out_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed_clean, f, indent=2)
        
    # 4. error_analysis.md
    with open(out_dir / "error_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Error Analysis — REAL GEMINI LOOCV\n\n")
        f.write("Detailed breakdown of incorrect predictions across Leave-One-Out Cross-Validation.\n\n")
        
        errors_found = False
        for r in loocv_results:
            failed_fields = [k for k, v in r["field_matches"].items() if not v]
            if failed_fields:
                errors_found = True
                f.write(f"## Document: `{r['file_name']}`\n")
                f.write("**Top Retrieved Examples**:\n")
                for ex in r["retrieved_examples"]:
                    f.write(f"- `{ex['file_name']}` (similarity: {ex['similarity']:.4f})\n")
                f.write("\n**Failed Fields**:\n")
                for k in failed_fields:
                    exp_val = r["ground_truth"].get(k)
                    pred_val = r["prediction"].get(k)
                    category = categorize_error(k, str(exp_val), str(pred_val), r["document_text"])
                    f.write(f"- **Field**: `{k}`\n")
                    f.write(f"  - **Expected**: `{exp_val}`\n")
                    f.write(f"  - **Predicted**: `{pred_val}`\n")
                    f.write(f"  - **Failure Category**: `{category}`\n\n")
        
        if not errors_found:
            f.write("Zero errors! Exact-match recall is 100% across all fields.\n")
            
    print(f"\nSaved LOOCV artifacts to {out_dir}")


if __name__ == "__main__":
    main()

