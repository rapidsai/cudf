# Copyright (c) 2018-2019, NVIDIA CORPORATION.

from cudf import dataframe
from cudf import datasets
from cudf.dataframe import DataFrame, from_pandas, merge
from cudf.dataframe import Index, MultiIndex
from cudf.dataframe import Series
from cudf.multi import concat
from cudf.io import (read_avro, read_csv, read_feather, read_hdf, read_json,
                     read_orc, read_parquet, from_dlpack)
from cudf.settings import set_options
from cudf.reshape import melt
from cudf.ops import (sqrt, sin, cos, tan, arcsin, arccos, arctan, exp, log,
                      logical_not, logical_and, logical_or)

from librmm_cffi import librmm as rmm
from cudf.reshape import get_dummies

# Versioneer
from cudf._version import get_versions
__version__ = get_versions()['version']
del get_versions
