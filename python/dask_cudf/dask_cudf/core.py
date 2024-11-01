# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import warnings

# This module provides backward compatibility for
# users who have yet to migrate to query-planning
from dask_cudf import (
    Index,  # noqa: F401
    Series,  # noqa: F401
    groupby_agg,  # noqa: F401
    read_text,  # noqa: F401
    to_orc,  # noqa: F401
)

warnings.warn(
    "The `dask_cudf.core` module is now deprecated. "
    "Please import from the top-level `dask_cudf` module only.",
    FutureWarning,
)
