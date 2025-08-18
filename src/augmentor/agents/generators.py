from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ..core.openrouter import OpenRouterClient
from loguru import logger
from tqdm import tqdm


@dataclass
class GeneratedItem:
    prompt: str
    source: str  # type: paraphrase|conditional|syntax|brainstorm
    intent_disc: int
    intent_cont: Optional[float] = None


def _clean_text(out: str) -> str:
    t = out.strip()
    t = t.strip("`\"' \n")
    t = re.sub(r"^\s*[\-\*\d\.\)]\s+", "", t)
    t = re.sub(r"(?i)^prompt\s*:\s*", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


async def paraphrase_batch(
    client: OpenRouterClient,
    seeds: List[Tuple[str, int]],
    model: str,
    temperature: float,
    similarity_threshold: float,
    seed_intent_cont: Optional[Dict[str, float]] = None,
) -> List[GeneratedItem]:
    # Simple prompt with constraint mention; actual validation is in validator
    async def one(seed: Tuple[str, int]) -> GeneratedItem:
        text, cls = seed
        messages = [
            {
                "role": "system",
                "content": "Rephrase the user prompt preserving meaning. Return exactly one line, no quotes, no bullets.",
            },
            {
                "role": "user",
                "content": f"Prompt: {text}\nConstraint: keep high semantic similarity (cosine >= {similarity_threshold}). Output only the new prompt.",
            },
        ]
        out = await client.chat(messages, model=model, temperature=temperature)
        cont = seed_intent_cont.get(text) if seed_intent_cont else None
        return GeneratedItem(prompt=_clean_text(out), source="paraphrase", intent_disc=cls, intent_cont=cont)

    if not seeds:
        return []

    tasks = [asyncio.create_task(one(s)) for s in seeds]
    items: List[GeneratedItem] = []
    with tqdm(total=len(tasks), desc="Paraphrases", unit="it", leave=True, dynamic_ncols=True) as pbar:
        for t in asyncio.as_completed(tasks):
            try:
                res = await t
                items.append(res)
            except BaseException as e:
                logger.warning("paraphrase_batch: one failed: %s", repr(e))
            finally:
                pbar.update(1)
    return items


async def conditional_generate(
    client: OpenRouterClient, cls: int, k: int, model: str, temperature: float, class_intent_cont: Optional[Dict[int, float]] = None
) -> List[GeneratedItem]:
    async def one() -> GeneratedItem:
        messages = [
            {"role": "system", "content": "Create a realistic user prompt for a general-purpose assistant."},
            {
                "role": "user",
                "content": f"Target intent class: {cls}.\nReturn exactly one plausible prompt. No quotes, no lists.",
            },
        ]
        out = await client.chat(messages, model=model, temperature=temperature)
        cont = class_intent_cont.get(int(cls)) if class_intent_cont else None
        return GeneratedItem(prompt=_clean_text(out), source="conditional", intent_disc=cls, intent_cont=cont)

    if k <= 0:
        return []

    tasks = [asyncio.create_task(one()) for _ in range(k)]
    items: List[GeneratedItem] = []
    with tqdm(total=len(tasks), desc=f"Conditional c{cls}", unit="it", leave=True, dynamic_ncols=True) as pbar:
        for t in asyncio.as_completed(tasks):
            try:
                res = await t
                items.append(res)
            except BaseException as e:
                logger.warning("conditional_generate: one failed: %s", repr(e))
            finally:
                pbar.update(1)
    return items


async def syntax_transform(
    client: OpenRouterClient, seeds: List[Tuple[str, int]], model: str, temperature: float, seed_intent_cont: Optional[Dict[str, float]] = None
) -> List[GeneratedItem]:
    async def one(seed: Tuple[str, int]) -> GeneratedItem:
        text, cls = seed
        messages = [
            {
                "role": "system",
                "content": "Rewrite using a different grammatical structure (active/passive, clause reordering).",
            },
            {"role": "user", "content": f"Text: {text}\nReturn exactly one transformed sentence. No quotes."},
        ]
        out = await client.chat(messages, model=model, temperature=temperature)
        cont = seed_intent_cont.get(text) if seed_intent_cont else None
        return GeneratedItem(prompt=_clean_text(out), source="syntax", intent_disc=cls, intent_cont=cont)

    if not seeds:
        return []

    tasks = [asyncio.create_task(one(s)) for s in seeds]
    items: List[GeneratedItem] = []
    with tqdm(total=len(tasks), desc="Syntax", unit="it", leave=True, dynamic_ncols=True) as pbar:
        for t in asyncio.as_completed(tasks):
            try:
                res = await t
                items.append(res)
            except BaseException as e:
                logger.warning("syntax_transform: one failed: %s", repr(e))
            finally:
                pbar.update(1)
    return items


async def brainstorm(
    client: OpenRouterClient,
    k: int,
    model: str,
    temperature: float,
    quotas: Dict[int, int],
    class_intent_cont: Optional[Dict[int, float]] = None,
) -> List[GeneratedItem]:
    if k <= 0:
        return []
    classes = list(quotas.keys())
    weights = [max(1, quotas[c]) for c in classes]

    def pick_class() -> int:
        return random.choices(classes, weights=weights, k=1)[0]

    async def one() -> GeneratedItem:
        cls = pick_class()
        messages = [
            {
                "role": "system",
                "content": "Invent a novel, safe and useful user prompt unlike obvious near-duplicates.",
            },
            {"role": "user", "content": "Return exactly one prompt. No quotes, no lists."},
        ]
        out = await client.chat(messages, model=model, temperature=temperature)
        cont = class_intent_cont.get(int(cls)) if class_intent_cont else None
        return GeneratedItem(prompt=_clean_text(out), source="brainstorm", intent_disc=cls, intent_cont=cont)

    tasks = [asyncio.create_task(one()) for _ in range(k)]
    items: List[GeneratedItem] = []
    with tqdm(total=len(tasks), desc="Brainstorm", unit="it", leave=True, dynamic_ncols=True) as pbar:
        for t in asyncio.as_completed(tasks):
            try:
                res = await t
                items.append(res)
            except BaseException as e:
                logger.warning("brainstorm: one failed: %s", repr(e))
            finally:
                pbar.update(1)
    return items
