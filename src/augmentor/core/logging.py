from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from loguru import logger


def setup_logging(verbose: bool = False, logs_dir: str | Path = "logs") -> Path:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logfile = Path(logs_dir) / f"run_{ts}.log"

    logger.remove()
    logger.add(lambda m: print(m, end=""))  # stdout
    logger.add(str(logfile), enqueue=True)

    if verbose:
        logger.level("DEBUG")
    else:
        logger.level("INFO")

    logger.info(f"Logs → {logfile}")
    return logfile
