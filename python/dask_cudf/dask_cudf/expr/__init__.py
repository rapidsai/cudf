# Copyright (c) 2024, NVIDIA CORPORATION.

from dask import config

# Check if dask-dataframe is using dask-expr
QUERY_PLANNING_ON = config.get("dataframe.query-planning", False)

# Register custom expressions and collections
try:
    import dask_cudf.expr._collection
    import dask_cudf.expr._expr

    DASK_EXPR_ENABLED = True  # Dask-expr is installed
except ImportError as err:
    DASK_EXPR_ENABLED = False  # Dask-expr is not installed
    if QUERY_PLANNING_ON:
        # Dask *should* raise an error before this.
        # However, we can still raise here to be certain.
        raise RuntimeError(
            "Failed to register the 'cudf' backend for dask-expr."
            " Please make sure you have dask-expr installed.\n"
            f"Error Message: {err}"
        )
