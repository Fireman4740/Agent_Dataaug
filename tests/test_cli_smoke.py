import json
from pathlib import Path

from augmentor.core.config import AppConfig
from augmentor.graph import run_workflow

import pytest


class DummyClient:
    async def chat(self, *args, **kwargs):
        return "This is a test prompt"


@pytest.mark.asyncio
async def test_run_workflow_smoke(tmp_path: Path, monkeypatch):
    # Prepare a tiny dataset
    p = tmp_path / "data.jsonl"
    rows = [
        {"prompt": "Hello", "dataset": "x", "intent_disc": 0, "intent_cont": 0.1},
        {"prompt": "World", "dataset": "x", "intent_disc": 1, "intent_cont": 0.2},
    ]
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cfg = AppConfig()
    # Bypass API key
    monkeypatch.setenv(cfg.openrouter.api_key_env, "test-key")

    # Monkeypatch OpenRouterClient within graph to use DummyClient
    from augmentor import graph as graph_mod
    graph_mod.OpenRouterClient = lambda *a, **k: DummyClient()

    out = await run_workflow(p, cfg)
    assert out.augmented
    assert isinstance(out.report, str)
