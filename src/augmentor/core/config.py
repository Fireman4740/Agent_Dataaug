from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any
import yaml


@dataclass
class OpenRouterConfig:
    api_key_env: str = "OPENROUTER_API_KEY"
    models: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "creative": ["openai/gpt-4o-mini", "anthropic/claude-3-opus"],
            "analysis": ["openai/gpt-4o-mini"],
        }
    )


@dataclass
class GenerationConfig:
    target_coverage: int = 100
    max_new_per_class: int = 2000
    similarity_threshold: float = 0.82
    diversity_temperature: float = 0.9
    max_similarity: float = 0.95
    max_brainstorm_ratio: float = 0.1
    examples_per_class: int = 3


@dataclass
class RateLimitConfig:
    max_calls_per_minute: int = 60
    max_concurrency: int = 5


@dataclass
class AppConfig:
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    @staticmethod
    def load(path: str | None) -> "AppConfig":
        if path is None:
            return AppConfig()
        with open(path, "r") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        # Merge with defaults
        cfg = AppConfig()
        if "openrouter" in data:
            orc = data["openrouter"] or {}
            cfg.openrouter.api_key_env = orc.get("api_key_env", cfg.openrouter.api_key_env)
            cfg.openrouter.models = orc.get("models", cfg.openrouter.models)
        if "generation" in data:
            gc = data["generation"] or {}
            cfg.generation.target_coverage = gc.get("target_coverage", cfg.generation.target_coverage)
            cfg.generation.max_new_per_class = gc.get("max_new_per_class", cfg.generation.max_new_per_class)
            cfg.generation.similarity_threshold = gc.get("similarity_threshold", cfg.generation.similarity_threshold)
            cfg.generation.diversity_temperature = gc.get("diversity_temperature", cfg.generation.diversity_temperature)
            cfg.generation.max_similarity = gc.get("max_similarity", cfg.generation.max_similarity)
            cfg.generation.max_brainstorm_ratio = gc.get("max_brainstorm_ratio", cfg.generation.max_brainstorm_ratio)
            cfg.generation.examples_per_class = gc.get("examples_per_class", cfg.generation.examples_per_class)
        if "rate_limit" in data:
            rc = data["rate_limit"] or {}
            cfg.rate_limit.max_calls_per_minute = rc.get("max_calls_per_minute", cfg.rate_limit.max_calls_per_minute)
            cfg.rate_limit.max_concurrency = rc.get("max_concurrency", cfg.rate_limit.max_concurrency)
        return cfg

    def get_api_key(self) -> str:
        key = os.getenv(self.openrouter.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"Missing OpenRouter API key in env var {self.openrouter.api_key_env}"
            )
        return key
