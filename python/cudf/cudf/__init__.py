# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from cudf.utils.gpu_utils import validate_setup  # isort:skip

validate_setup(check_dask=False)

import cupy
from numba import cuda

import rmm

from cudf import core, datasets
from cudf._version import get_versions
from cudf.core import (
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Float32Index,
    Float64Index,
    Index,
    Int8Index,
    Int16Index,
    Int32Index,
    Int64Index,
    MultiIndex,
    RangeIndex,
    Series,
    UInt8Index,
    UInt16Index,
    UInt32Index,
    UInt64Index,
    from_pandas,
    merge,
)
from cudf.core.dtypes import CategoricalDtype
from cudf.core.groupby import Grouper
from cudf.core.ops import (
    arccos,
    arcsin,
    arctan,
    cos,
    exp,
    floor_divide,
    log,
    logical_and,
    logical_not,
    logical_or,
    remainder,
    sin,
    sqrt,
    tan,
)
from cudf.core.reshape import concat, get_dummies, melt, merge_sorted
from cudf.core.series import isclose
from cudf.core.tools.datetimes import to_datetime
from cudf.io import (
    from_dlpack,
    read_avro,
    read_csv,
    read_feather,
    read_hdf,
    read_json,
    read_orc,
    read_parquet,
)
from cudf.utils.utils import set_allocator

cuda.set_memory_manager(rmm.RMMNumbaManager)
cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

__version__ = get_versions()["version"]
del get_versions
