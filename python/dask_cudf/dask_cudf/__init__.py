# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from dask import config

# For dask>2024.2.0, we can silence the loud deprecation
# warning before importing `dask.dataframe` (this won't
# do anything for dask==2024.2.0)
config.set({"dataframe.query-planning-warning": False})

import dask.dataframe as dd
from dask.dataframe import from_delayed

import cudf

from . import backends
from ._version import __git_commit__, __version__
from .core import concat, from_cudf, from_dask_dataframe
from .expr import QUERY_PLANNING_ON


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
    from .expr._collection import DataFrame, Index, Series

    groupby_agg = raise_not_implemented_error("groupby_agg")
    read_text = DataFrame.read_text
    to_orc = raise_not_implemented_error("to_orc")

else:
    from .core import DataFrame, Index, Series
    from .groupby import groupby_agg
    from .io import read_text, to_orc


__all__ = [
    "DataFrame",
    "Series",
    "Index",
    "from_cudf",
    "from_dask_dataframe",
    "concat",
    "from_delayed",
]


if not hasattr(cudf.DataFrame, "mean"):
    cudf.DataFrame.mean = None
del cudf
