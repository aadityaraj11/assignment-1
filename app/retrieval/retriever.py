"""
app/retrieval/retriever.py
==========================
Handles indexing and retrieving semantic examples.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.retrieval.training_data import TrainingExample

log = logging.getLogger(__name__)

class RetrieverIndex:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: list[dict] = []
        
    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        emb_path = self.index_dir / "embeddings.npy"
        meta_path = self.index_dir / "metadata.json"
        
        if not emb_path.exists() or not meta_path.exists():
            log.warning("Index not found at %s", self.index_dir)
            return False
            
        try:
            self.embeddings = np.load(emb_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            log.info("Loaded index with %d examples", len(self.metadata))
            return True
        except Exception as e:
            log.error("Failed to load index: %s", e)
            return False

    def save(self, embeddings: np.ndarray, metadata: list[dict]):
        """Save index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        emb_path = self.index_dir / "embeddings.npy"
        meta_path = self.index_dir / "metadata.json"
        
        np.save(emb_path, embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        self.embeddings = embeddings
        self.metadata = metadata
        log.info("Saved index with %d examples to %s", len(metadata), self.index_dir)

    def search(self, query_emb: np.ndarray, top_k: int = 3) -> list[tuple[TrainingExample, float]]:
        """Retrieve top_k similar training examples."""
        if self.embeddings is None or len(self.metadata) == 0:
            log.warning("Search called on empty index.")
            return []
            
        # Ensure query is 2D
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
            
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        
        # Get top-k indices descending
        best_idx = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in best_idx:
            meta = self.metadata[idx]
            score = float(sims[idx])
            ex = TrainingExample(
                file_name=meta["file_name"],
                document_text=meta["document_text"],
                metadata=meta["metadata"]
            )
            results.append((ex, score))
            
        return results
