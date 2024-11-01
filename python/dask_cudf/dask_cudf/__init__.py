# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from dask import config

import dask.dataframe as dd
from dask.dataframe import from_delayed  # noqa: E402

import cudf  # noqa: E402

from . import backends  # noqa: E402, F401
from ._version import __git_commit__, __version__  # noqa: E402, F401
from .core import concat, from_cudf, DataFrame, Index, Series  # noqa: F401

QUERY_PLANNING_ON = dd.DASK_EXPR_ENABLED


def read_csv(*args, **kwargs):
    with config.set({"dataframe.backend": "cudf"}):
        return dd.read_csv(*args, **kwargs)


def read_json(*args, **kwargs):
    with config.set({"dataframe.backend": "cudf"}):
        return dd.read_json(*args, **kwargs)


def read_orc(*args, **kwargs):
    with config.set({"dataframe.backend": "cudf"}):
        return dd.read_orc(*args, **kwargs)


def read_parquet(*args, **kwargs):
    with config.set({"dataframe.backend": "cudf"}):
        return dd.read_parquet(*args, **kwargs)


def raise_not_implemented_error(attr_name):
    def inner_func(*args, **kwargs):
        raise NotImplementedError(
            f"Top-level {attr_name} API is not available for dask-expr."
        )

    return inner_func


if QUERY_PLANNING_ON:
    from ._expr.expr import _patch_dask_expr

    groupby_agg = raise_not_implemented_error("groupby_agg")
    read_text = DataFrame.read_text
    to_orc = raise_not_implemented_error("to_orc")
    _patch_dask_expr()

else:
    from ._legacy.groupby import groupby_agg  # noqa: F401
    from ._legacy.io import read_text, to_orc  # noqa: F401


__all__ = [
    "DataFrame",
    "Series",
    "Index",
    "from_cudf",
    "concat",
    "from_delayed",
]


if not hasattr(cudf.DataFrame, "mean"):
    cudf.DataFrame.mean = None
del cudf
