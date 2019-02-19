# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf import dataframe

from cudf.dataframe import DataFrame, from_pandas, merge
from cudf.dataframe import Index
from cudf.dataframe import Series
from cudf.multi import concat
from cudf.io import (read_csv, read_parquet, read_feather, read_json,
                     read_hdf, read_orc)
from cudf.settings import set_options
from cudf.reshape import melt


# Versioneer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
