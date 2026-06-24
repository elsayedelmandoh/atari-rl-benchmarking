"""propose: configuration package -- settings, logger, json config loading.
input: .env file, configs/*.json.
output: settings singleton, logger, loaded config dicts."""

from src.config.config import settings
from src.config.logger import logger

__all__ = ["settings", "logger"]
