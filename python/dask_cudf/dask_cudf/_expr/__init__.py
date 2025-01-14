# Copyright (c) 2024-2025, NVIDIA CORPORATION.

try:
    from dask.dataframe import dask_expr  # noqa: F401

except ImportError:
    # TODO: Remove when pinned to dask>2024.12.1
    import dask.dataframe as dd

    if not dd._dask_expr_enabled():
        raise ValueError(
            "The legacy DataFrame API is not supported for RAPIDS >24.12. "
            "The 'dataframe.query-planning' config must be True or None."
        )
