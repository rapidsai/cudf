# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import dask.dataframe as dd
from dask import config
from dask.dataframe import from_delayed

import cudf

from . import backends
from ._version import __git_commit__, __version__
from .core import concat, from_cudf, from_dask_dataframe
from .expr import DASK_EXPR_ENABLED


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


if DASK_EXPR_ENABLED:
    __all__ = [
        "from_cudf",
        "from_dask_dataframe",
        "concat",
        "from_delayed",
    ]
else:
    from .core import DataFrame, Series
    from .groupby import groupby_agg
    from .io import read_text, to_orc

    __all__ = [
        "DataFrame",
        "Series",
        "from_cudf",
        "from_dask_dataframe",
        "concat",
        "from_delayed",
    ]

if not hasattr(cudf.DataFrame, "mean"):
    cudf.DataFrame.mean = None
del cudf
