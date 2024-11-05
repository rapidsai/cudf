# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import warnings
from importlib import import_module

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


def _deprecated_api(old_api, new_api=None, rec=None):
    def inner_func(*args, **kwargs):
        if new_api:
            # Use alternative
            msg = f"{old_api} is now deprecated. "
            msg += rec or f"Please use {new_api} instead."
            warnings.warn(msg, FutureWarning)
            new_attr = new_api.split(".")
            module = import_module(".".join(new_attr[:-1]))
            return getattr(module, new_attr[-1])(*args, **kwargs)

        # No alternative - raise an error
        raise NotImplementedError(
            f"{old_api} is no longer supported. " + (rec or "")
        )

    return inner_func


if QUERY_PLANNING_ON:
    from ._expr.expr import _patch_dask_expr
    from . import io  # noqa: F401

    groupby_agg = _deprecated_api("dask_cudf.groupby_agg")
    read_text = DataFrame.read_text
    _patch_dask_expr()

else:
    from ._legacy.groupby import groupby_agg  # noqa: F401
    from ._legacy.io import read_text  # noqa: F401
    from . import io  # noqa: F401


to_orc = _deprecated_api(
    "dask_cudf.to_orc",
    new_api="dask_cudf._legacy.io.to_orc",
    rec="Please use DataFrame.to_orc instead.",
)


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
