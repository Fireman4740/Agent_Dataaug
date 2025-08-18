# augmentor-cli

Command-line tool to balance and expand a prompt dataset via a LangGraph-like multi-agent workflow using OpenRouter-hosted LLMs.

## Quick start

1. Python 3.11
2. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

3. Configure

- Export your OpenRouter key:

```bash
export OPENROUTER_API_KEY=...
```

- Adjust `config.yaml` if needed

4. Run

```bash
augmentor-cli --config config.yaml --new-runs 3 --verbose
# or
python main.py --config config.yaml --new-runs 3 --verbose
```

Input: `intent_mapped_with_elo.jsonl` (JSONL with fields: prompt, dataset, intent_disc, intent_cont)

Outputs: `augmented.jsonl`, `report.md`, `logs/run_<ts>.log`

## Architecture (ASCII)

```
CLI -> Orchestrator
   analyze -> plan -> [paraphrase, conditional, syntax, brainstorm] -> validate -> dedup -> sink
```

## Notes

- Embeddings: sentence-transformers all-MiniLM-L6-v2 (set `AUGMENTOR_FAKE_EMBEDDINGS=1` to use a fast deterministic embedder for tests/CI)
- Quality gates enforced at end (see report.md)
- Tests use mocked OpenRouter responses
