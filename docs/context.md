# Context Document for Prompt Dataset Augmentation Tool
Use environment variable `OPENROUTER_API_KEY` for authentication.
Always log progress to stdout and to `logs/run_<timestamp>.log`.
Terminate with exit code 0 only if the quality gates all pass.
Context Document (detailed project specification)
1. High-Level Goal
Create a command-line tool that automatically balances and enlarges a prompt dataset without fine-tuning any model.
The tool relies on a set of collaborating LLM agents, each accessible through the OpenRouter API. LangGraph is used to coordinate the flow of tasks between agents.

1. Input / Output
• Input: intent_mapped_with_elo.jsonl – each line contains a JSON object:

json
Copy
{"prompt": "...", "dataset": "...", "intent_disc": 0-4, "intent_cont": 0.00-1.00}
• Output files

augmented.jsonl – only the newly generated prompts.
report.md – coverage statistics, class histograms, quality metrics, and a chronological log of agent actions.
Log directory: logs/run_<timestamp>.log.
3. Runtime Configuration
All settings live in config.yaml and/or can be overridden via flags, e.g.:

yaml
Copy
openrouter:
  api_key_env: OPENROUTER_API_KEY
  models:
    creative: ["openai/gpt-4o-mini", "anthropic/claude-3-opus"]
    analysis: ["openai/gpt-4o-mini"]

generation:
  target_coverage: 100          # % of rows mapped to a balanced set
  max_new_per_class: 2000
  similarity_threshold: 0.82    # cosine on SBERT embeddings
  diversity_temperature: 0.9
rate_limit:
  max_calls_per_minute: 60
CLI Example

bash
Copy
python main.py --config config.yaml --new-runs 3 --verbose
4. Architectural Overview
┌───────────────────────────────┐
│            CLI                │
└──────────────┬────────────────┘
               │
        LangGraph Orchestrator
               │
   ┌───────────┴───────────┐
   │                       │
Dataset-Analyzer      Planner / Scheduler
   │                       │
   ├──────►  Coverage report│
   │                       │
   ▼                       ▼
Generator Agents      Quality Control Agents
(Paraphraser,           (Semantic-Validator,
 Conditional,            Duplicate-Checker)
 Syntax, Brainstorm)          │
   │                           │
   └───────────────merge───────┘
               │
        Sink: `augmented.jsonl`
5. Agent Responsibilities
Dataset-Analyzer
• Compute class counts, lexical diversity, n-gram entropy, cluster prompts with UMAP + HDBSCAN.
• Identify under-represented intent_disc values and prepare generation quotas.
Paraphraser
• Prompt engineering template keeps semantic cosine ≥ 0.82 while altering surface form.
• Uses beam search (k=5), selects the top-2 diverse beams.
Conditional-Generator
• Creates brand-new prompts for classes below coverage target.
• Two-stage generation: (concept sentence → final prompt).
Syntax-Augmenter
• Applies grammatical transformations through LLM instructions (passive↔active, clause re-ordering).
Creative-Brainstormer
• Explores «white-space» concepts: runs latent-space walk to find prompts unlike any current cluster.
• Introduces up to 10 % of total new prompts outside existing classes.
Semantic-Validator
• SBERT similarity to source (if paraphrase) OR plausibility heuristics (if novel).
• Rejects outputs failing language quality checks (grammar, profanity).
Duplicate-Checker
• MinHash + exact hash on normalized text; removes near duplicates among new and old data.
6. Quality Gates
A batch is accepted only if:
• Overall mapping coverage ≥ 90 %.
• New class histogram variance ≤ 5 %.
• Mean SBERT similarity (for paraphrases) in [0.82, 0.95].
• ≤ 1 % duplicates inside augmented.jsonl.

Failures trigger automated back-off and regeneration attempts (max 3).
If still failing, exit with code 1 and leave a diagnostic section in report.md.

7. Implementation Notes
• Language & Libs: Python 3.11, LangGraph, pandas, scikit-learn, sentence-transformers, t-qdm, PyYAML, loguru.
• Concurrency: asyncio tasks per agent; OpenRouter calls wrapped in tenacity.retry.
• Embedding Cache: TinyDB or SQLite storing text→vector to avoid redundant calls.
• Testing: PyTest with mocked OpenRouter responses; golden sample of 100 prompts.
• Packaging: Ship as a pip-installable module augmentor, entry-point augmentor-cli.
• CI: GitHub Actions running lint, unit tests, and an integration smoke test.

8. Execution Flow (pseudo-code)
python
Copy
graph = LangGraph()
graph.add_node("analyze", analyze_dataset)
graph.add_node("plan", make_plan)
graph.add_subgraph("generate", [paraphrase, conditional, syntax, brainstorm])
graph.add_node("validate", validate_batch)
graph.add_node("dedup", deduplicate)
graph.add_node("sink", write_outputs)

graph.add_edges([
    ("analyze", "plan"),
    ("plan", "generate"),
    ("generate", "validate"),
    ("validate", "dedup"),
    ("dedup", "sink")
])

graph.run(initial_state)
9. Deliverables
Source code repository (/src, /tests, /docs).
Default config.yaml.
Quick-start README describing environment setup and CLI usage.
Example report.md generated from a dry-run on a 1 000-row sample.