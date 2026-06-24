"""compat shim -- re-export from config package.
prefer from src.config.loader import ... directly."""

from src.config.loader import config_hash, load_all_configs, load_json  # noqa: F401
