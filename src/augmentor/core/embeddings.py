from __future__ import annotations

from functools import lru_cache
from typing import List
import os
import hashlib
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional import for unit tests
    SentenceTransformer = None  # type: ignore


class _FakeEmbedder:
    """Deterministic fake embedder for tests/CI, avoids model download.

    Produces 384-dim unit vectors derived from a stable hash of the text.
    """

    dim = 384

    def encode(self, texts: List[str], convert_to_numpy=True, normalize_embeddings=True):  # noqa: D401
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            # expand hash deterministically into dim floats
            arr = np.frombuffer(h * (self.dim // len(h) + 1), dtype=np.uint8)[: self.dim]
            v = arr.astype(np.float32)
            if normalize_embeddings:
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
            vecs.append(v)
        return np.stack(vecs)


@lru_cache(maxsize=1)
def get_embedder():
    # Use fake embedder if requested (e.g., in CI)
    if os.getenv("AUGMENTOR_FAKE_EMBEDDINGS", "").strip() == "1":
        return _FakeEmbedder()
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    # Correct model ID for SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: List[str]):
    model = get_embedder()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
