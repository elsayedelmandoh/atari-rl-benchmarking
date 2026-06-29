"""propose: cross-cutting shared utilities -- device selection, seeding, dependency checks.
input: user-supplied parameters (seed, device preference).
output: configured torch device, seeded rng, dependency status."""

from __future__ import annotations

import importlib
import random
import sys
from dataclasses import dataclass

import numpy as np

# -- device helpers --


def select_torch_device(preferred: str = "auto"):
    import torch

    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def device_summary() -> str:
    import torch

    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return f"cuda available: {torch.version.cuda}; devices={names}"
    return f"cuda unavailable in torch build; torch={torch.__version__}; cuda={torch.version.cuda}"


# -- seeding --


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        pass


# -- dependency checks --


@dataclass
class DependencyStatus:
    module: str
    installed: bool
    version: str | None
    error: str | None = None


def check_module(module: str) -> DependencyStatus:
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", None)
        return DependencyStatus(module=module, installed=True, version=version)
    except Exception as exc:
        return DependencyStatus(module=module, installed=False, version=None, error=str(exc))


def runtime_summary() -> str:
    return f"{sys.executable} | Python {sys.version.split()[0]}"


def check_required_modules() -> list[DependencyStatus]:
    return [
        check_module("numpy"),
        check_module("gymnasium"),
        check_module("ale_py"),
        check_module("torch"),
        check_module("stable_baselines3"),
        check_module("PIL"),
    ]
