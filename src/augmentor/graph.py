from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TypedDict, cast

from loguru import logger
import numpy as np
from tqdm import tqdm

from .core.config import AppConfig
from .core.openrouter import OpenRouterClient
from .agents.analyzer import analyze_dataset, load_dataset
from .agents.planner import make_plan
from .agents.generators import (
    paraphrase_batch,
    conditional_generate,
    syntax_transform,
    brainstorm,
    GeneratedItem,
)
from .agents.validator import (
    deduplicate,
    filter_by_similarity,
    build_signature_set,
    deduplicate_against,
    pairwise_similarity,
)

# LangGraph
from langgraph.graph import StateGraph, START, END


@dataclass
class RunOutputs:
    augmented: List[Dict]
    report: str
    metrics: Dict[str, float]


class WorkflowState(TypedDict, total=False):
    analysis: Any
    df_len: int
    df_prompts: List[str]
    seeds: List[Tuple[str, int]]
    model_analysis: str
    model_creative: str
    plan: Any
    client: Any
    paraphrases: List[GeneratedItem]
    paraphrase_pairs: List[Tuple[str, str]]
    paraphrase_sims: List[float]
    synthetics: List[GeneratedItem]
    cond_all: List[GeneratedItem]
    brainstorms: List[GeneratedItem]
    augmented: List[Dict]
    report: str
    metrics: Dict[str, float]
    class_intent_cont: Dict[int, float]
    seed_intent_cont: Dict[str, float]


def _class_histogram_variation(intents: List[int]) -> float:
    if not intents:
        return 0.0
    counts: Dict[int, int] = {}
    for c in intents:
        counts[c] = counts.get(c, 0) + 1
    arr = np.array(list(counts.values()), dtype=float)
    if arr.mean() == 0:
        return 0.0
    cv_percent = float(arr.std(ddof=0) / arr.mean() * 100.0)
    return cv_percent


def _sample_seeds_by_class(df, per_class: int) -> List[Tuple[str, int]]:
    seeds: List[Tuple[str, int]] = []
    for cls, group in df.groupby("intent_disc"):
        if len(group) > per_class:
            g = group.sample(n=per_class, random_state=42)
        else:
            g = group
        seeds.extend(list(zip(g["prompt"].astype(str).tolist(), g["intent_disc"].astype(int).tolist())))
    return seeds


def _req(state: WorkflowState, key: str) -> Any:
    v = state.get(key)
    if v is None:
        raise KeyError(f"Missing required state key: {key}")
    return v


async def run_workflow(input_path: str | Path, cfg: AppConfig) -> RunOutputs:
    """Exécute le workflow via LangGraph et renvoie les sorties de run."""

    async def node_analyze(state: WorkflowState) -> WorkflowState:
        logger.info("Analyzing dataset...")
        analysis = analyze_dataset(input_path)
        df = load_dataset(input_path)
        seeds = _sample_seeds_by_class(df, per_class=50)
        seeds = seeds[: min(200, len(seeds))]
        # intent_cont mappings
        class_intent_cont: Dict[int, float] = {}
        seed_intent_cont: Dict[str, float] = {}
        if "intent_cont" in df.columns:
            try:
                tmp = df[["prompt", "intent_disc", "intent_cont"]].dropna()
                tmp["intent_cont"] = tmp["intent_cont"].astype(float)
                for cls_val, grp in tmp.groupby("intent_disc"):
                    # Obtenir un int Python natif pour la classe
                    try:
                        cls_int = int(np.asarray(grp["intent_disc"].iloc[0]).item())
                    except Exception:
                        cls_int = int(str(cls_val))
                    vals = [float(v) for v in grp["intent_cont"].tolist()]
                    if vals:
                        m = np.mean(vals)
                        class_intent_cont[cls_int] = float(m.item() if hasattr(m, "item") else m)
                # map seed prompt -> intent_cont from original si présent
                src_map = {str(r["prompt"]): float(r["intent_cont"]) for _, r in tmp.iterrows()}
                for text, _ in seeds:
                    if text in src_map:
                        seed_intent_cont[text] = src_map[text]
            except Exception as e:
                logger.warning("Failed computing intent_cont maps: {}", repr(e))
        state.update(
            {
                "analysis": analysis,
                "df_len": len(df),
                "df_prompts": df["prompt"].astype(str).tolist(),
                "seeds": seeds,
                "model_analysis": cfg.openrouter.models["analysis"][0],
                "model_creative": cfg.openrouter.models["creative"][0],
                "class_intent_cont": class_intent_cont,
                "seed_intent_cont": seed_intent_cont,
            }
        )
        return state

    async def node_plan(state: WorkflowState) -> WorkflowState:
        analysis = _req(state, "analysis")
        plan = make_plan(
            cast(Any, analysis).class_counts,
            cfg.generation.target_coverage,
            cfg.generation.max_new_per_class,
        )
        # Client OpenRouter avec sémaphore de rate-limit simple
        api_key = cfg.get_api_key()
        sem = asyncio.Semaphore(max(1, cfg.rate_limit.max_concurrency))
        client = OpenRouterClient(
            api_key,
            semaphore=sem,
            calls_per_minute=cfg.rate_limit.max_calls_per_minute,
        )
        state.update({"plan": plan, "client": client})
        return state

    async def node_paraphrase(state: WorkflowState) -> WorkflowState:
        seeds: List[Tuple[str, int]] = state.get("seeds", [])
        if not seeds:
            state["paraphrases"] = []
            state["paraphrase_pairs"] = []
            state["paraphrase_sims"] = []
            return state
        client = _req(state, "client")
        model_analysis = _req(state, "model_analysis")
        items = await paraphrase_batch(
            cast(Any, client),
            seeds,
            model=cast(str, model_analysis),
            temperature=0.7,
            similarity_threshold=cfg.generation.similarity_threshold,
            seed_intent_cont=cast(Dict[str, float], state.get("seed_intent_cont", {})),
        )
        pairs_raw = [(s, p.prompt) for (s, _), p in zip(seeds, items)]
        kept_pairs, keep_idx, kept_sims = filter_by_similarity(
            pairs_raw,
            min_sim=cfg.generation.similarity_threshold,
            max_sim=cfg.generation.max_similarity,
        )
        state["paraphrases"] = [items[i] for i in keep_idx]
        state["paraphrase_pairs"] = kept_pairs
        state["paraphrase_sims"] = kept_sims
        return state

    async def node_syntax(state: WorkflowState) -> WorkflowState:
        seeds: List[Tuple[str, int]] = state.get("seeds", [])
        if not seeds:
            state["synthetics"] = []
            return state
        client = _req(state, "client")
        model_analysis = _req(state, "model_analysis")
        items = await syntax_transform(
            cast(Any, client),
            seeds,
            model=cast(str, model_analysis),
            temperature=0.6,
            seed_intent_cont=cast(Dict[str, float], state.get("seed_intent_cont", {})),
        )
        state["synthetics"] = items
        return state

    async def node_conditional(state: WorkflowState) -> WorkflowState:
        plan = _req(state, "plan")
        client = _req(state, "client")
        model_creative = _req(state, "model_creative")
        cond_all: List[GeneratedItem] = []
        quotas = cast(Any, plan).quotas
        total = int(sum(int(v) for v in quotas.values()))
        class_cont = cast(Dict[int, float], state.get("class_intent_cont", {}))
        if total <= 0:
            state["cond_all"] = []
            return state
        with tqdm(total=total, desc="Conditional (all)", unit="it", leave=True, dynamic_ncols=True) as pbar:
            for cls, k in quotas.items():
                if k > 0:
                    batch = await conditional_generate(
                        cast(Any, client),
                        int(cls),
                        int(k),
                        model=cast(str, model_creative),
                        temperature=cfg.generation.diversity_temperature,
                        class_intent_cont=class_cont,
                    )
                    cond_all.extend(batch)
                    pbar.update(len(batch))
        state["cond_all"] = cond_all
        return state

    async def node_brainstorm(state: WorkflowState) -> WorkflowState:
        plan = _req(state, "plan")
        client = _req(state, "client")
        model_creative = _req(state, "model_creative")
        planned_total = int(sum(cast(Any, plan).quotas.values()))
        k = min(
            max(10, len(state.get("seeds", [])) // 10),
            int(cfg.generation.max_brainstorm_ratio * max(planned_total, 1)),
        )
        class_cont = cast(Dict[int, float], state.get("class_intent_cont", {}))
        items = await brainstorm(
            cast(Any, client),
            k=k,
            model=cast(str, model_creative),
            temperature=0.9,
            quotas=cast(Any, plan).quotas,
            class_intent_cont=class_cont,
        )
        state["brainstorms"] = items
        return state

    async def node_validate_dedup(state: WorkflowState) -> WorkflowState:
        paraphrases: List[GeneratedItem] = state.get("paraphrases", [])
        synthetics: List[GeneratedItem] = state.get("synthetics", [])
        cond_all: List[GeneratedItem] = state.get("cond_all", [])
        brainstorms: List[GeneratedItem] = state.get("brainstorms", [])

        all_items = paraphrases + synthetics + cond_all + brainstorms

        kept_sims = state.get("paraphrase_sims", [])
        if kept_sims:
            mean_sim = float(np.mean(kept_sims))
        else:
            pairs = state.get("paraphrase_pairs", [])
            mean_sim = float(np.mean(pairwise_similarity(pairs))) if pairs else 1.0

        texts = [it.prompt for it in all_items]
        unique_texts, dup_ratio_internal = deduplicate(texts)

        source_sigs = build_signature_set(state.get("df_prompts", []))
        unique_texts, dup_ratio_vs_source = deduplicate_against(unique_texts, source_sigs)

        class_intent_cont = cast(Dict[int, float], state.get("class_intent_cont", {}))
        text_to_class: Dict[str, int] = {}
        text_to_cont: Dict[str, Optional[float]] = {}
        for it in all_items:
            if it.prompt not in text_to_class:
                cls_i = int(it.intent_disc)
                text_to_class[it.prompt] = cls_i
                cont_val = it.intent_cont if it.intent_cont is not None else class_intent_cont.get(cls_i)
                text_to_cont[it.prompt] = cont_val

        augmented = [
            {
                "prompt": t,
                "dataset": "augmented",
                "intent_disc": text_to_class.get(t, 0),
                "intent_cont": text_to_cont.get(t),
            }
            for t in unique_texts
        ]

        plan = _req(state, "plan")
        analysis = _req(state, "analysis")
        planned_new = float(sum(cast(Any, plan).quotas.values()))
        generated_unique = float(len(unique_texts))
        coverage = float(100.0 * (generated_unique / planned_new)) if planned_new > 0 else 100.0
        class_variation = _class_histogram_variation([row["intent_disc"] for row in augmented])

        report = (
            f"Coverage target: {cfg.generation.target_coverage}%\n"
            f"Original rows: {state.get('df_len', 0)}\n"
            f"Planned new: {int(planned_new)}\n"
            f"Generated total: {len(all_items)} (unique {len(unique_texts)})\n"
            f"Coverage achieved: {coverage:.2f}%\n"
            f"Paraphrase mean cosine: {mean_sim:.3f}\n"
            f"Duplicate ratio (internal): {dup_ratio_internal:.3f}\n"
            f"Duplicate ratio (vs source): {dup_ratio_vs_source:.3f}\n"
            f"Class histogram CV%: {class_variation:.2f}\n"
            f"Class counts (orig): {cast(Any, analysis).class_counts}\n"
        )

        metrics = {
            "coverage_percent": coverage,
            "paraphrase_mean_similarity": float(mean_sim),
            "duplicate_ratio": float(dup_ratio_internal),
            "class_histogram_cv_percent": float(class_variation),
        }

        state.update({"augmented": augmented, "report": report, "metrics": metrics})
        return state

    # Construction du graphe
    graph = StateGraph(WorkflowState)
    graph.add_node("analyze", node_analyze)
    graph.add_node("plan", node_plan)
    graph.add_node("paraphrase", node_paraphrase)
    graph.add_node("syntax", node_syntax)
    graph.add_node("conditional", node_conditional)
    graph.add_node("brainstorm", node_brainstorm)
    graph.add_node("validate_dedup", node_validate_dedup)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "paraphrase")
    graph.add_edge("paraphrase", "syntax")
    graph.add_edge("syntax", "conditional")
    graph.add_edge("conditional", "brainstorm")
    graph.add_edge("brainstorm", "validate_dedup")
    graph.add_edge("validate_dedup", END)

    app = graph.compile()

    final_state = cast(WorkflowState, await app.ainvoke({}))

    return RunOutputs(
        augmented=final_state.get("augmented", []),
        report=final_state.get("report", ""),
        metrics=final_state.get("metrics", {}),
    )
