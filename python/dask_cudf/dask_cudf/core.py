# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import dask.dataframe as dd

# This module provides backward compatibility for legacy import patterns.
if dd.DASK_EXPR_ENABLED:
    from dask_cudf._expr.collection import (  # noqa: E402
        DataFrame,
        Index,
        Series,
    )
else:
    from dask_cudf._legacy.core import DataFrame, Index, Series  # noqa: F401

from dask_cudf._legacy.core import concat, from_cudf  # noqa: F401
