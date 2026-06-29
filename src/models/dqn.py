"""propose: dqn algorithm setup -- configuration and model factory.
input: environment, hyperparameters, seed.
output: configured sb3 dqn model."""

from __future__ import annotations

from typing import Any

from src.models.sb3 import get_sb3_class, normalize_sb3_params


def make_dqn(env: Any, seed: int, params: dict[str, Any]) -> Any:
    model_cls = get_sb3_class("DQN")
    return model_cls(
        "CnnPolicy",
        env,
        seed=seed,
        verbose=0,
        device="auto",
        tensorboard_log="./logs",
        **normalize_sb3_params(params),
    )
