"""compat shim -- re-export from src.models.discrete_sac.
prefer from src.models.discrete_sac import ... directly."""

from src.models.discrete_sac import (  # noqa: F401
    DiscreteSAC,
    NatureBackbone,
    ReplayBatch,
    ReplayBuffer,
    build_policy,
)
