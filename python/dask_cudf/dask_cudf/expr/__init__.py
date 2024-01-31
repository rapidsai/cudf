# Copyright (c) 2024, NVIDIA CORPORATION.

# Make sure custom expressions and collections are defined
try:
    import dask_cudf.expr._collection
    import dask_cudf.expr._expr

    _expr_support = True
except ImportError:
    # Dask Expressions not installed
    _expr_support = False
