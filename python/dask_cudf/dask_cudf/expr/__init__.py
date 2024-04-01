# Copyright (c) 2024, NVIDIA CORPORATION.

from dask import config

# Check if dask-dataframe is using dask-expr.
# For dask>=2024.3.0, a null value will default to True
QUERY_PLANNING_ON = config.get("dataframe.query-planning", None) is not False

# Register custom expressions and collections
if QUERY_PLANNING_ON:
    try:
        import dask_cudf.expr._collection
        import dask_cudf.expr._expr

    except ImportError as err:
        # Dask *should* raise an error before this.
        # However, we can still raise here to be certain.
        raise RuntimeError(
            "Failed to register the 'cudf' backend for dask-expr."
            " Please make sure you have dask-expr installed.\n"
            f"Error Message: {err}"
        )
