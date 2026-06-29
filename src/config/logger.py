"""propose: preconfigured file + console logger for the project.
input: logger name, optional log file path.
output: configured logger instance."""

import logging
import sys
from pathlib import Path

from src.config.config import settings


def get_logger(
    name: str = "atari-rl",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is None:
        log_file = settings.LOGS_DIR / "project.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()
