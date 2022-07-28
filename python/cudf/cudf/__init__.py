# Copyright (c) 2018-2022, NVIDIA CORPORATION.

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
from cudf.core.scalar import Scalar

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
from cudf.core.dataframe import DataFrame, from_pandas, merge, from_dataframe
from cudf.core.series import Series
from cudf.core.missing import NA
from cudf.core.multiindex import MultiIndex
from cudf.core.cut import cut
from cudf.core.algorithms import factorize
from cudf.core.dtypes import (
    CategoricalDtype,
    Decimal64Dtype,
    Decimal32Dtype,
    Decimal128Dtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.core.groupby import Grouper
from cudf.core.reshape import (
    concat,
    get_dummies,
    melt,
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

from cudf.options import (
    get_option,
    set_option,
    describe_option,
)

try:
    from ptxcompiler.patch import patch_numba_codegen_if_needed
except ImportError:
    pass
else:
    # Patch Numba to support CUDA enhanced compatibility.
    # See https://github.com/rapidsai/ptxcompiler for
    # details.
    patch_numba_codegen_if_needed()
    del patch_numba_codegen_if_needed

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

__all__ = [
    "BaseIndex",
    "CategoricalDtype",
    "CategoricalIndex",
    "DataFrame",
    "DateOffset",
    "DatetimeIndex",
    "Decimal32Dtype",
    "Decimal64Dtype",
    "Float32Index",
    "Float64Index",
    "GenericIndex",
    "Grouper",
    "Index",
    "Int16Index",
    "Int32Index",
    "Int64Index",
    "Int8Index",
    "IntervalDtype",
    "IntervalIndex",
    "ListDtype",
    "MultiIndex",
    "NA",
    "RangeIndex",
    "Scalar",
    "Series",
    "StringIndex",
    "StructDtype",
    "TimedeltaIndex",
    "UInt16Index",
    "UInt32Index",
    "UInt64Index",
    "UInt8Index",
    "api",
    "concat",
    "cut",
    "date_range",
    "describe_option",
    "factorize",
    "from_dataframe",
    "from_dlpack",
    "from_pandas",
    "get_dummies",
    "get_option",
    "interval_range",
    "isclose",
    "melt",
    "merge",
    "pivot",
    "read_avro",
    "read_csv",
    "read_feather",
    "read_hdf",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_text",
    "set_allocator",
    "set_option",
    "testing",
    "to_datetime",
    "to_numeric",
    "unstack",
]
