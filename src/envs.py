"""compat shim -- re-export from utils.data_acquisition.
prefer from src.utils.data_acquisition import ... directly."""

from src.utils.data_acquisition import (  # noqa: F401
    import_gymnasium,
    make_atari_env,
    make_raw_env,
    make_vec_atari_env,
    resolve_all_envs,
    resolve_env_id,
)
