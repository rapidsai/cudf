# Copyright (c) 2018-2021, NVIDIA CORPORATION.
from cudf.utils.gpu_utils import validate_setup  # isort:skip

validate_setup()

import cupy
from numba import cuda

import rmm

from cudf import core, datasets, testing
from cudf._version import get_versions
from cudf.api.extensions import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)
from cudf.core import (
    NA,
    BaseIndex,
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
    IntervalIndex,
    MultiIndex,
    RangeIndex,
    Scalar,
    Series,
    TimedeltaIndex,
    UInt8Index,
    UInt16Index,
    UInt32Index,
    UInt64Index,
    cut,
    from_pandas,
    interval_range,
    merge,
)
from cudf.core.algorithms import factorize
from cudf.core.dtypes import (
    CategoricalDtype,
    Decimal64Dtype,
    Decimal32Dtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
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
from cudf.core.tools.datetimes import DateOffset, to_datetime
from cudf.core.tools.numeric import to_numeric
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
