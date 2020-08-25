# Copyright (c) 2018-2020, NVIDIA CORPORATION.
from cudf.utils.gpu_utils import validate_setup  # isort:skip

validate_setup()

import cupy
from numba import cuda

import rmm

import cudf.api.types
from cudf import core, datasets, testing
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
    TimedeltaIndex,
    UInt8Index,
    UInt16Index,
    UInt32Index,
    UInt64Index,
    from_pandas,
    merge,
)
from cudf.core.dtypes import (
    dtype,
    Generic,
    Datetime,
    Floating,
    Number,
    Integer,
    Flexible,
    Datetime,
    Timedelta,
    CategoricalDtype, 
    Int8Dtype,
    Int16Dtype, 
    Int32Dtype, 
    Int64Dtype, 
    UInt8Dtype, 
    UInt16Dtype,
    UInt32Dtype, 
    UInt64Dtype, 
    StringDtype,
    Float32Dtype,
    Float64Dtype, 
    BooleanDtype,
    Datetime64NSDtype,
    Datetime64USDtype, 
    Datetime64MSDtype,
    Datetime64SDtype,
    Timedelta64NSDtype,
    Timedelta64USDtype,
    Timedelta64MSDtype,
    Timedelta64SDtype
)

from cudf.core.groupby import Grouper
from cudf.core.ops import (
    add,
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
    multiply,
    remainder,
    sin,
    sqrt,
    subtract,
    tan,
    true_divide,
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
from cudf.utils.dtypes import _NA_REP
from cudf.utils.utils import set_allocator

cuda.set_memory_manager(rmm.RMMNumbaManager)
cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

__version__ = get_versions()["version"]
del get_versions
