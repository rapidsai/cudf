# Copyright (c) 2018-2023, NVIDIA CORPORATION.

from dask.dataframe import from_delayed

import cudf

from . import backends
from ._version import __git_commit__, __version__
from .core import DataFrame, Series, concat, from_cudf, from_dask_dataframe
from .groupby import groupby_agg
from .io import read_csv, read_json, read_orc, read_text, to_orc

try:
    from .io import read_parquet
except ImportError:
    pass

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
