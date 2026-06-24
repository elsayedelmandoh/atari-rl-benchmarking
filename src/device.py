"""compat shim -- re-export from utils.helpers.
prefer from src.utils.helpers import ... directly."""

from src.utils.helpers import device_summary, select_torch_device  # noqa: F401
