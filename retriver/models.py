"""
Model Singletons
=================
Lazy-loaded bi-encoder and cross-encoder so they are instantiated once
and reused across all retrieve() calls.
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from .config import BIENCODER_MODEL, CROSSENCODER_MODEL

_BIENCODER    = None
_CROSSENCODER = None


def get_biencoder() -> SentenceTransformer:
    """Return (and cache) the bi-encoder used for Stage 1 fusion scoring."""
    global _BIENCODER
    if _BIENCODER is None:
        _BIENCODER = SentenceTransformer(BIENCODER_MODEL)
    return _BIENCODER


def get_crossencoder() -> CrossEncoder:
    """Return (and cache) the cross-encoder used for Stage 2 reranking
    and Stage 4.5 column pruning."""
    global _CROSSENCODER
    if _CROSSENCODER is None:
        _CROSSENCODER = CrossEncoder(CROSSENCODER_MODEL)
    return _CROSSENCODER
