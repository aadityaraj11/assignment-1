"""
scripts/build_index.py
======================
Builds the retrieval index using Phase 2 loaders and sentence-transformers.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.loaders import load_document
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.retriever import RetrieverIndex
from app.retrieval.training_data import load_training_metadata, find_training_file

DATA_DIR = PROJECT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TRAIN_CSV = DATA_DIR / "train.csv"
INDEX_DIR = PROJECT_DIR / "artifacts" / "retrieval_index"

def main():
    parser = argparse.ArgumentParser(description="Build semantic retrieval index.")
    parser.parse_args()
    
    print("=" * 80)
    print(" BUILDING SEMANTIC RETRIEVAL INDEX")
    print("=" * 80)
    
    if not TRAIN_CSV.exists():
        sys.exit(f"Training CSV not found: {TRAIN_CSV}")
        
    print(f"Loading metadata from {TRAIN_CSV} ...")
    metadata_map = load_training_metadata(TRAIN_CSV)
    
    print("\nLoading Embedding Model ...")
    embedder = EmbeddingModel()
    
    embeddings = []
    metadata_list = []
    
    print("\nProcessing Training Documents ...")
    for stem, target_meta in metadata_map.items():
        fpath = find_training_file(TRAIN_DIR, stem)
        if not fpath:
            print(f" [!] Missing file for '{stem}'")
            continue
            
        print(f" [+] Ingesting {fpath.name}")
        doc = load_document(fpath)
        
        # Ensure we have some text to embed
        text = doc.normalized_text
        if not text.strip():
            print(f" [!] Empty text for {fpath.name}, skipping.")
            continue
            
        emb = embedder.encode(text)
        embeddings.append(emb)
        
        metadata_list.append({
            "file_name": stem,
            "document_text": text,
            "metadata": target_meta
        })
        
    if not embeddings:
        sys.exit("No documents successfully processed.")
        
    import numpy as np
    emb_matrix = np.vstack(embeddings)
    
    index = RetrieverIndex(INDEX_DIR)
    index.save(emb_matrix, metadata_list)
    
    print("\n" + "=" * 80)
    print(" INDEX BUILD COMPLETE")
    print("=" * 80)
    print(f" Documents indexed : {len(metadata_list)}")
    print(f" Embedding model   : {embedder.model_name}")
    print(f" Embedding dim     : {emb_matrix.shape[1]}")
    print(f" Index path        : {INDEX_DIR}")
    
if __name__ == "__main__":
    main()
