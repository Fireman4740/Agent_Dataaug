from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from loguru import logger

from .core.config import AppConfig
from .core.logging import setup_logging
from .graph import run_workflow

from dotenv import load_dotenv


EXIT_OK = 0
EXIT_FAIL = 1


def parse_args():
    p = argparse.ArgumentParser(description="augmentor-cli")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--input", type=str, default="intent_mapped_with_elo.jsonl")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--new-runs", type=int, default=1)
    return p.parse_args()


def gates_pass(metrics) -> bool:
    return (
        metrics.get("coverage_percent", 0.0) >= 90.0
        and metrics.get("class_histogram_cv_percent", 100.0) <= 5.0
        and 0.82 <= metrics.get("paraphrase_mean_similarity", 0.0) <= 0.95
        and metrics.get("duplicate_ratio", 1.0) <= 0.01
    )


async def main_async():
    # Charge variables d'environnement depuis .env si présent
    load_dotenv()

    args = parse_args()
    setup_logging(verbose=bool(args.verbose))
    cfg = AppConfig.load(args.config)

    outputs_all = []
    for _ in range(args.new_runs):
        outputs = await run_workflow(args.input, cfg)
        outputs_all.append(outputs)

    # Merge results
    augmented = [row for out in outputs_all for row in out.augmented]
    report = "\n---\n".join(out.report for out in outputs_all)

    # Write outputs
    out_dir = Path(".")
    with open(out_dir / "augmented.jsonl", "w") as f:
        for row in augmented:
            f.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "report.md", "w") as f:
        f.write(report)

    # Distribution plot des prompts générés en fonction de intent_cont
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(augmented)
        # Ne garder que les lignes générées (dataset == 'augmented') et intent_cont non nul
        df = df[(df.get("dataset") == "augmented") & df["intent_cont"].notna()]
        if not df.empty:
            plt.figure(figsize=(8, 4))
            # Histogramme/ KDE par classe si dispo
            ax = df.groupby("intent_disc")["intent_cont"].plot(kind="kde", legend=True)
            plt.title("Distribution de intent_cont par classe (prompts générés)")
            plt.xlabel("intent_cont")
            plt.ylabel("densité")
            plt.tight_layout()
            plt.savefig(out_dir / "intent_cont_distribution.png", dpi=150)
            logger.info("Graphique enregistré: intent_cont_distribution.png")
        else:
            logger.warning("Aucune donnée avec intent_cont pour tracer la distribution.")
    except Exception as e:
        logger.warning(f"Impossible de générer le graphique de distribution: {e}")

    # Evaluate gates across the last run's metrics
    ok = gates_pass(outputs_all[-1].metrics)
    exit_code = EXIT_OK if ok else EXIT_FAIL
    logger.info(f"Finished with exit code {exit_code}")
    raise SystemExit(exit_code)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
