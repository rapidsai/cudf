# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import dask.dataframe as dd
from dask import config
from dask.dataframe import from_delayed

import cudf

from . import backends, io  # noqa: F401
from ._expr import collection  # noqa: F401
from ._expr.expr import _patch_dask_expr
from ._version import __git_commit__, __version__  # noqa: F401
from .core import DataFrame, Index, Series, _deprecated_api, concat, from_cudf


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


groupby_agg = _deprecated_api("dask_cudf.groupby_agg")
read_text = DataFrame.read_text
to_orc = _deprecated_api(
    "dask_cudf.to_orc",
    new_api="dask_cudf.io.orc.to_orc",
    rec="Please use DataFrame.to_orc instead.",
)


_patch_dask_expr()


__all__ = [
    "DataFrame",
    "Index",
    "Series",
    "concat",
    "from_cudf",
    "from_delayed",
]


if not hasattr(cudf.DataFrame, "mean"):
    cudf.DataFrame.mean = None
del cudf
