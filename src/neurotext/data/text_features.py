"""Text embeddings for sentences, with no downloads and no network.

TF-IDF followed by a truncated SVD (latent semantic analysis). This is a deliberately
simple, deterministic stand-in: it captures which words a sentence uses, not its
meaning. The loaders return `text_emb`, so a stronger frozen sentence encoder can replace
this function without touching any other code.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_svd_embeddings(sentences: list[str], dim: int = 64, seed: int = 0) -> np.ndarray:
    """(S, dim) float32 unit-norm embeddings; row i belongs to sentences[i]."""
    if len(sentences) < 3:
        raise ValueError("need at least 3 sentences")
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=1).fit_transform(sentences)
    dim = min(dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if dim < 1:
        raise ValueError("vocabulary too small to embed")
    emb = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(tfidf)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / np.maximum(norms, 1e-12)).astype(np.float32)
