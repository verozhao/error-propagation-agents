"""Severity metrics for error injection.

Primary: severity_semantic (1 - cosine on BGE-large embeddings)
Secondary: severity_edit (normalized character edit distance)
"""
import numpy as np

_ENCODER = None
_ENCODERS = {}

def get_encoder(model_name="BAAI/bge-large-en-v1.5"):
    """Lazy-load encoder. Cached globally."""
    global _ENCODERS
    if model_name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer
        _ENCODERS[model_name] = SentenceTransformer(model_name)
    return _ENCODERS[model_name]

def severity_semantic(pre_text: str, post_text: str, encoder_name: str = "BAAI/bge-large-en-v1.5") -> float:
    """Primary severity: 1 - cosine similarity on embeddings. Range [0, 1]."""
    if not pre_text or not post_text:
        return 0.0
    enc = get_encoder(encoder_name)
    embs = enc.encode([pre_text, post_text], normalize_embeddings=True)
    cosine = float(np.dot(embs[0], embs[1]))
    return round(max(0.0, 1.0 - cosine), 6)

def severity_edit(pre_text: str, post_text: str) -> float:
    """Secondary severity: normalized character edit distance. Range [0, 1]."""
    if not pre_text and not post_text:
        return 0.0
    max_len = max(len(pre_text), len(post_text), 1)
    diff = sum(1 for a, b in zip(pre_text, post_text) if a != b) + abs(len(pre_text) - len(post_text))
    return round(diff / max_len, 4)

def severity_multi_encoder(pre_text: str, post_text: str) -> dict:
    """Compute severity with all three encoders for robustness check."""
    encoders = [
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    results = {}
    for enc_name in encoders:
        short_name = enc_name.split("/")[-1]
        results[short_name] = severity_semantic(pre_text, post_text, encoder_name=enc_name)
    return results
