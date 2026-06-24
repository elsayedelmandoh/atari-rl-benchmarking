"""propose: validate model input contracts -- shape, layout, contiguity checks.
input: observation from environment, algorithm contract config.
output: observation metadata dict if all checks pass."""  # noqa: D400

from __future__ import annotations

from typing import Any

from src.evaluation.metadata import observation_metadata, shape_of


def assert_model_input_contract(algo: str, obs: Any, configs: dict) -> dict:
    contract = configs["contracts"]["contracts"][algo]
    expected = contract["expected_unbatched_shape"]
    if not isinstance(expected, list):
        raise ValueError(f"{algo} input contract is not approved yet: {expected}")
    actual = shape_of(obs)
    if tuple(expected) != actual:
        raise ValueError(f"{algo} expected input shape {tuple(expected)}, got {actual}")
    meta = observation_metadata(obs)
    if meta["layout"] != "CHW":
        raise ValueError(f"{algo} expected CHW unbatched layout, got {meta}")
    if meta["contiguous"] != "True":
        raise ValueError(f"{algo} expected contiguous observation memory, got {meta}")
    return meta
