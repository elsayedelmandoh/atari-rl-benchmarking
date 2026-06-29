"""propose: utility package -- data acquisition, processing, eda, and shared helpers.
input: environment configs, raw data, user parameters.
output: initialized environments, processed data, analysis helpers."""

from src.utils.data_acquisition import (
    import_gymnasium,
    make_atari_env,
    make_raw_env,
    make_vec_atari_env,
    resolve_all_envs,
    resolve_env_id,
)

__all__ = [
    "import_gymnasium",
    "make_atari_env",
    "make_raw_env",
    "make_vec_atari_env",
    "resolve_all_envs",
    "resolve_env_id",
]
