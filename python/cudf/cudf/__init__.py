# Copyright (c) 2018-2021, NVIDIA CORPORATION.
from cudf.utils.gpu_utils import validate_setup

validate_setup()

import cupy
from numba import config as numba_config, cuda

import rmm

from cudf.api.types import dtype
from cudf import api, core, datasets, testing
from cudf._version import get_versions
from cudf.api.extensions import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)
from cudf.core.scalar import (
    NA,
    Scalar,
)
from cudf.core.index import (
    BaseIndex,
    CategoricalIndex,
    DatetimeIndex,
    Float32Index,
    Float64Index,
    Index,
    GenericIndex,
    Int8Index,
    Int16Index,
    Int32Index,
    Int64Index,
    IntervalIndex,
    RangeIndex,
    StringIndex,
    TimedeltaIndex,
    UInt8Index,
    UInt16Index,
    UInt32Index,
    UInt64Index,
    interval_range,
)
from cudf.core.dataframe import DataFrame, from_pandas, merge
from cudf.core.series import Series
from cudf.core.multiindex import MultiIndex
from cudf.core.cut import cut
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
from cudf.core.reshape import (
    concat,
    get_dummies,
    melt,
    merge_sorted,
    pivot,
    unstack,
)
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
    read_text,
)
from cudf.core.tools.datetimes import date_range
from cudf.utils.dtypes import _NA_REP
from cudf.utils.utils import set_allocator

cuda.set_memory_manager(rmm.RMMNumbaManager)
cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

try:
    # Numba 0.54: Disable low occupancy warnings
    numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
except AttributeError:
    # Numba < 0.54: No occupancy warnings
    pass
del numba_config

__version__ = get_versions()["version"]
del get_versions
