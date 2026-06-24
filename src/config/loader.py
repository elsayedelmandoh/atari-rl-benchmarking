"""propose: load json config files from configs/ directory.
input: configs/*.json files on disk.
output: dict of parsed configuration."""

import hashlib
import json
from pathlib import Path
from typing import Any

from src.config.config import settings


def load_json(path: str | Path) -> dict[str, Any]:
    full_path = Path(path)
    if not full_path.is_absolute():
        full_path = settings.CONFIGS_DIR / full_path
    with full_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_configs() -> dict[str, Any]:
    return {
        "envs": load_json("envs.json"),
        "algorithms": load_json("algorithms.json"),
        "preprocessing": load_json("preprocessing.json"),
        "contracts": load_json("model_contracts.json"),
        "benchmark": load_json("benchmark.json"),
    }


def config_hash(configs: dict[str, Any]) -> str:
    payload = json.dumps(configs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
