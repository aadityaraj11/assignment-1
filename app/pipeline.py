import logging
from pathlib import Path
from typing import Optional

from app.loaders import load_document
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.retriever import RetrieverIndex
from app.schemas import MetadataExtraction
from app.llm.provider import get_provider
from app.llm.extractor import build_extraction_prompt

log = logging.getLogger(__name__)

# Constants relative to pipeline execution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INDEX_DIR = PROJECT_DIR / "artifacts" / "retrieval_index"

# Singleton instances initialized gracefully
_embedder: Optional[EmbeddingModel] = None
_index: Optional[RetrieverIndex] = None

def _get_embedder() -> EmbeddingModel:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingModel()
    return _embedder

def _get_index() -> RetrieverIndex:
    global _index
    if _index is None:
        _index = RetrieverIndex(INDEX_DIR)
        if not _index.load():
            log.warning("Could not load retrieval index. Proceeding without examples.")
    return _index

def extract_metadata(document_path: str | Path, artifacts_dir: Optional[Path] = None):
    """
    Core metadata extraction flow for Phase 3.
    Returns: (cleaned_prediction_dict, retrieved_examples, retrieval_scores, debug_info)
    """
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    # 1. Ingestion (Phase 2)
    doc_content = load_document(path, artifacts_dir=artifacts_dir)
    target_text = doc_content.normalized_text

    # 2. Embedding creation
    embedder = _get_embedder()
    query_emb = embedder.encode(target_text)

    # 3. Retrieval
    index = _get_index()
    retrieved_results = index.search(query_emb, top_k=3)
    examples = [ex for ex, score in retrieved_results]
    retrieval_scores = [(ex.file_name, float(score)) for ex, score in retrieved_results]

    # 4. LLM structured extraction
    prompt = build_extraction_prompt(target_text, examples)
    
    # Save prompt preview to artifacts/debug/example_prompt.txt for diagnostic auditing
    debug_dir = (artifacts_dir or (PROJECT_DIR / "artifacts")) / "debug"
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_dir / "example_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
    except Exception as e:
        log.warning("Could not write debug prompt file: %s", e)

    provider = get_provider()
    
    # Provider guarantees dictionary response based on MetadataExtraction structure
    raw_dict_prediction = provider.generate_structured(prompt, MetadataExtraction)
    
    # 5. Pydantic final validation enforcing explicit string schemas
    prediction = MetadataExtraction(**raw_dict_prediction)
    
    # 6. Minimal cleanup: strip whitespace to align string comparisons purely
    cleaned_dict = {}
    for k, v in prediction.model_dump().items():
        if isinstance(v, str):
            v_cleaned = v.strip()
            cleaned_dict[k] = v_cleaned if v_cleaned else None
        else:
            cleaned_dict[k] = v

    debug_info = {
        "prompt": prompt,
        "provider": provider.__class__.__name__,
        "model": getattr(provider, "model", "mock")
    }

    return cleaned_dict, examples, retrieval_scores, debug_info
