"""Shared Sentence-BERT embedder (used by RAG and the two-tower model) with a TF-IDF fallback."""
import threading

import numpy as np

from . import config

_lock = threading.Lock()
_model = None
_variant = None


def get_embedder():
    global _model, _variant
    with _lock:
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
                _variant = config.EMBED_MODEL.split("/")[-1]
            except Exception as e:  # noqa: BLE001
                print("[embedder] falling back to hashing TF-IDF:", e)
                _model = "hash"
                _variant = "hashing-tfidf (fallback)"
        return _model, _variant


def embed(texts, model=None) -> np.ndarray:
    model = model or get_embedder()[0]
    if model == "hash":
        from sklearn.feature_extraction.text import HashingVectorizer

        v = HashingVectorizer(n_features=2048, ngram_range=(1, 2), norm="l2", alternate_sign=False)
        return v.transform(texts).toarray().astype(np.float32)
    return model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=32)


def embedder_params():
    m, _ = get_embedder()
    if m == "hash":
        return 0, 0.0
    from .core import torch_param_stats

    s = torch_param_stats(m)
    return s["params"], s["param_mb"]
