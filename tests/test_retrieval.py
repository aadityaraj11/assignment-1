import pytest
import numpy as np
from pathlib import Path
import pandas as pd

from app.retrieval.training_data import _clean_val, load_training_metadata
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.retriever import RetrieverIndex

def test_clean_val():
    assert _clean_val("  hello  ") == "hello"
    assert _clean_val(np.nan) is None
    assert _clean_val(float("nan")) is None
    assert _clean_val(None) is None
    assert _clean_val(" ") is None
    assert _clean_val(60.0) == "60.0" # Usually int is handled in loop but raw testing clean

def test_load_training_metadata_missing_file():
    with pytest.raises(FileNotFoundError):
        load_training_metadata("non_existent.csv")

def test_retriever_index(tmp_path):
    index = RetrieverIndex(tmp_path)
    # mock data
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.707, 0.707]
    ])
    metadata = [
        {"file_name": "doc1", "document_text": "hello", "metadata": {}},
        {"file_name": "doc2", "document_text": "world", "metadata": {}},
        {"file_name": "doc3", "document_text": "both", "metadata": {}}
    ]
    
    # Save
    index.save(embeddings, metadata)
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "metadata.json").exists()
    
    # Load new instance
    index2 = RetrieverIndex(tmp_path)
    assert index2.load()
    assert index2.embeddings.shape == (3, 2)
    
    # Search for [1,0] should put doc1 first
    res = index2.search(np.array([1.0, 0.0]), top_k=2)
    assert len(res) == 2
    assert res[0][0].file_name == "doc1"
    # The score should be roughly 1.0
    assert pytest.approx(res[0][1], 0.01) == 1.0
