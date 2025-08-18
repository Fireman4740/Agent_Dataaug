from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class AnalysisResult:
    class_counts: Dict[int, int]
    lexical_diversity: float
    ngram_entropy: float
    total_rows: int


def load_dataset(path: str | Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "intent_disc" not in df.columns:
        raise ValueError("Missing 'intent_disc' in dataset")
    return df


def lexical_stats(prompts: List[str]) -> tuple[float, float]:
    # Type-Token Ratio
    tokens = [t.lower() for p in prompts for t in p.split()]
    types = set(tokens)
    ttr = len(types) / max(len(tokens), 1)

    # n-gram entropy (unigram)
    vec = CountVectorizer(ngram_range=(1, 1))
    X = vec.fit_transform(prompts)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    probs = freqs / freqs.sum() if freqs.sum() else freqs
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
    return ttr, float(entropy)


def analyze_dataset(path: str | Path) -> AnalysisResult:
    df = load_dataset(path)
    counts = df["intent_disc"].value_counts().to_dict()
    prompts = df["prompt"].astype(str).tolist()
    ttr, entropy = lexical_stats(prompts)
    return AnalysisResult(
        class_counts={int(k): int(v) for k, v in counts.items()},
        lexical_diversity=ttr,
        ngram_entropy=entropy,
        total_rows=len(df),
    )
