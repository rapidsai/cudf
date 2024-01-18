# Copyright (c) 2024, NVIDIA CORPORATION.

# Monkey-patch `Expr` with meta-based dispatching
from dask_cudf.expr_backend._dispatch_utils import patch_dask_expr

patch_dask_expr()

# Import the "cudf" backend
import dask_cudf.expr_backend._collection  # noqa: F401
import dask_cudf.expr_backend._expr  # noqa: F401
