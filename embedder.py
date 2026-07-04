"""Pluggable text embedder.

Production: sentence-transformers all-MiniLM-L6-v2 (matches the team's setup).
Testing:    a deterministic hash-based bag-of-words embedder so that metric
            calibration can run with zero heavy dependencies.

Both return unit-normalized vectors, which matters: VCV distances and cosine
similarities are only comparable across turns if every vector has norm 1.
"""
import hashlib
import re
import numpy as np


class MockEmbedder:
    """Deterministic bag-of-words random-projection embedder.

    Each token gets a fixed pseudo-random unit vector (seeded by the token's
    hash). A text's embedding is the normalized mean of its token vectors.
    Properties we rely on in tests:
      - identical texts -> cosine 1.0
      - texts with disjoint vocab -> cosine ~ 0
      - partially overlapping vocab -> intermediate cosine
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._cache = {}

    def _token_vec(self, tok: str) -> np.ndarray:
        v = self._cache.get(tok)
        if v is None:
            seed = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim)
            v /= np.linalg.norm(v)
            self._cache[tok] = v
        return v

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            toks = re.findall(r"[a-z0-9']+", t.lower())
            if not toks:
                out.append(np.zeros(self.dim))
                continue
            m = np.mean([self._token_vec(tok) for tok in toks], axis=0)
            n = np.linalg.norm(m)
            out.append(m / n if n > 0 else m)
        return np.array(out)


class SBERTEmbedder:
    """all-MiniLM-L6-v2, unit-normalized. Requires: pip install sentence-transformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)


def get_embedder(kind: str = "sbert"):
    return SBERTEmbedder() if kind == "sbert" else MockEmbedder()
