"""compat shim -- re-export from src.evaluation.metadata.
prefer from src.evaluation.metadata import ... directly."""

from src.evaluation.metadata import (  # noqa: F401
    dtype_of,
    infer_layout,
    is_contiguous,
    min_max,
    observation_metadata,
    shape_of,
)
