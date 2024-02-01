# Copyright (c) 2024, NVIDIA CORPORATION.

from dask import config

DASK_EXPR_ENABLED = False
if config.get("dataframe.query-planning", False):
    # Make sure custom expressions and collections are defined
    try:
        import dask_cudf.expr._collection
        import dask_cudf.expr._expr

        DASK_EXPR_ENABLED = True
    except ImportError:
        # Dask Expressions not installed.
        # Dask DataFrame should have already thrown an error
        # before we got here.
        pass
