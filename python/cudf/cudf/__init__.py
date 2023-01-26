# Copyright (c) 2018-2023, NVIDIA CORPORATION.

from cudf.utils.gpu_utils import validate_setup

validate_setup()

import cupy
from numba import config as numba_config, cuda

import rmm

from cudf import api, core, datasets, testing
from cudf._version import get_versions
from cudf.api.extensions import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)
from cudf.api.types import dtype
from cudf.core.algorithms import factorize
from cudf.core.cut import cut
from cudf.core.dataframe import DataFrame, from_dataframe, from_pandas, merge
from cudf.core.dtypes import (
    CategoricalDtype,
    Decimal32Dtype,
    Decimal64Dtype,
    Decimal128Dtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.core.groupby import Grouper
from cudf.core.index import (
    BaseIndex,
    CategoricalIndex,
    DatetimeIndex,
    Float32Index,
    Float64Index,
    GenericIndex,
    Index,
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
from cudf.core.missing import NA
from cudf.core.multiindex import MultiIndex
from cudf.core.reshape import (
    concat,
    crosstab,
    get_dummies,
    melt,
    pivot,
    pivot_table,
    unstack,
)
from cudf.core.scalar import Scalar
from cudf.core.series import Series, isclose
from cudf.core.tools.datetimes import DateOffset, date_range, to_datetime
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
from cudf.options import describe_option, get_option, set_option
from cudf.utils.dtypes import _NA_REP
from cudf.utils.utils import clear_cache, set_allocator

try:
    from cubinlinker.patch import patch_numba_linker_if_needed
except ImportError:
    pass
else:
    # Patch Numba to support CUDA enhanced compatibility.
    # cuDF requires a stronger set of conditions than what is
    # checked by patch_numba_linker_if_needed due to the PTX
    # files needed for JIT Groupby Apply and string UDFs
    from cudf.core.udf.groupby_utils import dev_func_ptx
    from cudf.core.udf.utils import _setup_numba_linker

    _setup_numba_linker(dev_func_ptx)

    del patch_numba_linker_if_needed

cuda.set_memory_manager(rmm.RMMNumbaManager)
cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

try:
    # Numba 0.54: Disable low occupancy warnings
    numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
except AttributeError:
    # Numba < 0.54: No occupancy warnings
    pass
del numba_config


rmm.register_reinitialize_hook(clear_cache)


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
    "crosstab",
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
    "pivot_table",
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
