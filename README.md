# augmentor-cli

Outil en ligne de commande pour équilibrer et étendre un jeu de données de prompts via un workflow multi‑agents orchestré avec LangGraph et des LLMs accessibles via OpenRouter.

## Installation rapide

- Python 3.11 recommandé
- Installation (editable):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Configuration

- Clé OpenRouter via variable d’environnement (ou fichier `.env`):

```bash
export OPENROUTER_API_KEY=xxxx
```

ou créez un fichier `.env` à la racine:

```
OPENROUTER_API_KEY=xxxx
```

- Fichier `config.yaml` (extrait des options clés):

```yaml
openrouter:
  api_key_env: OPENROUTER_API_KEY
  models:
    creative: ["<model-creative>", "<fallback>"]
    analysis: ["<model-analysis>"]

generation:
  target_coverage: 100 # % des besoins couverts
  max_new_per_class: 2000 # plafond par classe
  similarity_threshold: 0.82 # min cosine paraphrases
  max_similarity: 0.95 # max cosine paraphrases
  diversity_temperature: 0.9 # température LLM
  max_brainstorm_ratio: 0.10 # brainstorming ≤ 10 %
  examples_per_class: 3 # (réservé évolutions)

rate_limit:
  max_calls_per_minute: 60
  max_concurrency: 5 # parallélisme
```

## Données d’entrée/sortie

- Entrée: `intent_mapped_with_elo.jsonl` (lignes JSON: `prompt`, `dataset`, `intent_disc`, `intent_cont`)
- Sorties:
  - `augmented.jsonl` (uniquement les nouvelles générations)
  - `combined.jsonl` (dataset original + nouvelles générations, dédupliqué sur `prompt` insensible à la casse)
  - `report.md`
  - `logs/run_<timestamp>.log`

## Exécution

```bash
augmentor-cli --config config.yaml --input intent_mapped_with_elo.jsonl --new-runs 1 --verbose
# ou
python -m augmentor.main --config config.yaml --input intent_mapped_with_elo.jsonl --new-runs 1 --verbose
```

### Arguments CLI

| Argument          | Défaut                         | Description                                                                                                                                             |
| ----------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--config <path>` | (auto)                         | Chemin du fichier YAML de configuration. S'il est omis, le chargeur tente `config.yaml` à la racine.                                                    |
| `--input <path>`  | `intent_mapped_with_elo.jsonl` | Fichier d'entrée (JSON Lines) contenant les prompts d'origine.                                                                                          |
| `--verbose`       | faux                           | Active un niveau de log détaillé (DEBUG). Sans ce flag: logs plus concis (INFO).                                                                        |
| `--new-runs <n>`  | `1`                            | Nombre de ré-exécutions complètes du workflow; les sorties sont concaténées puis fusionnées (dédup simples) dans `augmented.jsonl` et `combined.jsonl`. |

Notes:

- Code de sortie: 0 si les quality gates passent, 1 sinon.
- Les exécutions multiples (`--new-runs > 1`) permettent d'augmenter la diversité; surveillez néanmoins les métriques de similarité et de duplication dans `report.md`.
- Les variables d'environnement (ex: `OPENROUTER_API_KEY`) peuvent être chargées via un fichier `.env`.

Code de retour 0 si les quality gates passent:

- couverture ≥ 90 %
- variance (CV%) de l’histogramme de classes ≤ 5 %
- similarité paraphrase ∈ [0.82, 0.95]
- doublons ≤ 1 %

## Architecture (LangGraph)

```
START -> analyze -> plan -> paraphrase -> syntax -> conditional -> brainstorm -> validate_dedup -> END
```

- analyze: stats et classes
- plan: quotas par classe (scale par target_coverage)
- paraphrase: reformulations contrôlées par similarité
- syntax: transformations grammaticales
- conditional: génération conditionnelle par classe
- brainstorm: idées nouvelles plafonnées (≤ max_brainstorm_ratio)
- validate_dedup: filtrage par similarité, MinHash+LSH (datasketch) + dédup vs dataset

## Détails techniques

- Orchestrateur: LangGraph (async)
- LLM: OpenRouter (env `OPENROUTER_API_KEY`)
- Embeddings: sentence-transformers (mode CI possible avec `AUGMENTOR_FAKE_EMBEDDINGS=1`)
- Déduplication: datasketch (MinHash + LSH), fallback hashing exact
- Journalisation: loguru (stdout + fichier)

## Tests

```bash
pytest -q
```

Astuce CI: utilisez `AUGMENTOR_FAKE_EMBEDDINGS=1` pour éviter le téléchargement de modèles d’embeddings.
