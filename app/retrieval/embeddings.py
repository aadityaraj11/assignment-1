"""
app/retrieval/embeddings.py
===========================
Wraps semantic embedding model logic (sentence-transformers).
Provides context-aware chunking for very long documents before pooling.
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

# Lightweight highly reproducible model
DEFAULT_MODEL = "all-MiniLM-L6-v2"

class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is not installed.")

        log.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        
    def encode(self, text: str) -> np.ndarray:
        """
        Produce a dense semantic vector representation of the string.
        For simplicity on 10 docs, we chunk by 2000 chars, encode each, 
        and mean-pool to get document representation.
        """
        # Chunking conceptually
        chunk_size = 2000
        overlap = 200
        chunks = []
        
        if not text:
            text = "empty document"
            
        start = 0
        while start < len(text):
            chunks.append(text[start:start+chunk_size])
            start += (chunk_size - overlap)
            
        embeddings = self.encoder.encode(chunks, convert_to_numpy=True)
        # Average pooling handles document representation generally.
        pooled = np.mean(embeddings, axis=0)
        
        # normalize
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
            
        return pooled
