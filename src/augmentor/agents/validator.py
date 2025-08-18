from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Any, Set

import numpy as np

from ..core.embeddings import embed_texts

try:
    from datasketch import MinHash, MinHashLSH
except Exception:  # pragma: no cover
    MinHash = None  # type: ignore
    MinHashLSH = None  # type: ignore


@dataclass
class ValidationMetrics:
    mean_similarity: float
    duplicate_ratio: float


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T)


def pairwise_similarity(paraphrase_pairs: List[Tuple[str, str]]) -> List[float]:
    if not paraphrase_pairs:
        return []
    texts = [t for pair in paraphrase_pairs for t in pair]
    embs = embed_texts(texts)
    sims: List[float] = []
    for i in range(0, len(texts), 2):
        v1 = embs[i]
        v2 = embs[i + 1]
        sims.append(float(v1 @ v2))
    return sims


def filter_by_similarity(paraphrase_pairs: List[Tuple[str, str]], min_sim: float, max_sim: float) -> Tuple[List[Tuple[str, str]], List[int], List[float]]:
    sims = pairwise_similarity(paraphrase_pairs)
    keep_idx: List[int] = [i for i, s in enumerate(sims) if min_sim <= s <= max_sim]
    kept_pairs = [paraphrase_pairs[i] for i in keep_idx]
    kept_sims = [sims[i] for i in keep_idx]
    return kept_pairs, keep_idx, kept_sims


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def minhash_signature(text: str) -> str:
    normalized = _normalize(text)
    return hashlib.md5(normalized.encode()).hexdigest()


def _shingles(text: str, n: int = 3) -> List[str]:
    toks = _normalize(text).split()
    if len(toks) < n:
        return [" ".join(toks)] if toks else []
    return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def _mh_from_text(text: str, num_perm: int = 128) -> Any:
    m = MinHash(num_perm=num_perm)  # type: ignore[call-arg]
    for sh in _shingles(text, n=3):
        m.update(sh.encode("utf-8"))
    return m


def deduplicate(texts: Iterable[str], lsh_threshold: float = 0.9) -> Tuple[List[str], float]:
    texts_list = list(texts)
    if not texts_list:
        return [], 0.0

    # Fallback simple exact dedup if datasketch unavailable
    if MinHash is None or MinHashLSH is None:
        seen = set()
        out: List[str] = []
        for t in texts_list:
            sig = minhash_signature(t)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(t)
        dup_ratio = 1 - len(out) / max(len(texts_list), 1)
        return out, dup_ratio

    # LSH-based near-duplicate detection
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=128)  # type: ignore[call-arg]
    unique: List[str] = []
    exact_seen = set()

    for idx, t in enumerate(texts_list):
        norm = _normalize(t)
        if norm in exact_seen:
            continue
        mh = _mh_from_text(t)
        # query approximate matches among previously stored uniques
        if unique:
            matches = lsh.query(mh)  # type: ignore[attr-defined]
        else:
            matches = []
        if matches:
            # near-duplicate found; skip
            continue
        # accept and index
        key = f"k{len(unique)}"
        lsh.insert(key, mh)  # type: ignore[attr-defined]
        exact_seen.add(norm)
        unique.append(t)

    dup_ratio = 1 - len(unique) / max(len(texts_list), 1)
    return unique, float(dup_ratio)


def build_signature_set(texts: Iterable[str]) -> Set[str]:
    return {minhash_signature(t) for t in texts}


def deduplicate_against(texts: Iterable[str], existing_signatures: Set[str]) -> Tuple[List[str], float]:
    texts_list = list(texts)
    out: List[str] = []
    seen = set(existing_signatures)  # copy
    for t in texts_list:
        sig = minhash_signature(t)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(t)
    dup_ratio = 1 - len(out) / max(len(texts_list), 1)
    return out, dup_ratio
