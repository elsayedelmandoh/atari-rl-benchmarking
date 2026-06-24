"""propose: application entrypoint -- run full project pipeline.
input: .env configuration, configs/*.json.
output: initialized project structure, run summary."""

import sys

from src.config.config import settings
from src.config.logger import logger


def main() -> int:
    logger.info("project: %s", settings.PROJECT_NAME)
    logger.info("environment: %s", settings.ENV)
    logger.info("data dir: %s", settings.DATA_DIR)
    logger.info("artifacts dir: %s", settings.ARTIFACTS_DIR)
    logger.info("configs dir: %s", settings.CONFIGS_DIR)

    print(f"project: {settings.PROJECT_NAME}")
    print(f"environment: {settings.ENV}")
    print(f"data: {settings.DATA_DIR}")
    print(f"artifacts: {settings.ARTIFACTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
