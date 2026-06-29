"""propose: stable-baselines3 bridge -- shared utilities for sb3 algorithm setup.
input: hyperparameter dicts, progress values.
output: normalized params, sb3 class references."""

from __future__ import annotations

from typing import Any


def get_sb3_class(class_name: str):
    import stable_baselines3 as sb3

    return getattr(sb3, class_name)


def linear_schedule(initial_value: float):
    initial_value = float(initial_value)

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def normalize_sb3_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("lin_"):
            normalized[key] = linear_schedule(float(value.removeprefix("lin_")))
        else:
            normalized[key] = value
    return normalized
