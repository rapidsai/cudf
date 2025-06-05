# Copyright (c) 2018-2025, NVIDIA CORPORATION.

# If libcudf was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libcudf
except ModuleNotFoundError:
    pass
else:
    libcudf.load_library()
    del libcudf

# _setup_numba _must be called before numba.cuda is imported, because
# it sets the numba config variable responsible for enabling
# Minor Version Compatibility. Setting it after importing numba.cuda has no effect.
from cudf.utils._numba import _setup_numba
from cudf.utils.gpu_utils import validate_setup

_setup_numba()
validate_setup()

del _setup_numba
del validate_setup

import cupy
from numba import cuda

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.numba import RMMNumbaManager

from cudf import api, core, datasets, testing
from cudf._version import __git_commit__, __version__
from cudf.api.extensions import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)
from cudf.api.types import dtype
from cudf.core.algorithms import factorize, unique
from cudf.core.cut import cut
from cudf.core.dataframe import DataFrame, from_pandas, merge
from cudf.core.dtypes import (
    CategoricalDtype,
    Decimal32Dtype,
    Decimal64Dtype,
    Decimal128Dtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.core.groupby import Grouper, NamedAgg
from cudf.core.index import (
    CategoricalIndex,
    DatetimeIndex,
    Index,
    IntervalIndex,
    RangeIndex,
    TimedeltaIndex,
    interval_range,
)
from cudf.core.missing import NA, NaT
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
from cudf.options import (
    describe_option,
    get_option,
    option_context,
    set_option,
)

cuda.set_memory_manager(RMMNumbaManager)
cupy.cuda.set_allocator(rmm_cupy_allocator)

del cuda
del cupy
del rmm_cupy_allocator
del RMMNumbaManager

rmm.register_reinitialize_hook(lambda: Scalar._clear_instance_cache())

del rmm

__all__ = [
    "NA",
    "CategoricalDtype",
    "CategoricalIndex",
    "DataFrame",
    "DateOffset",
    "DatetimeIndex",
    "Decimal32Dtype",
    "Decimal64Dtype",
    "Decimal128Dtype",
    "Grouper",
    "Index",
    "IntervalDtype",
    "IntervalIndex",
    "ListDtype",
    "MultiIndex",
    "NaT",
    "NamedAgg",
    "RangeIndex",
    "Scalar",
    "Series",
    "StructDtype",
    "TimedeltaIndex",
    "api",
    "concat",
    "core",  # TODO: core should not be publicly exposed
    "crosstab",
    "cut",
    "datasets",
    "date_range",
    "describe_option",
    "dtype",  # TODO: dtype should not be a public function
    "errors",
    "factorize",
    "from_dlpack",
    "from_pandas",
    "get_dummies",
    "get_option",
    "interval_range",
    "io",
    "isclose",
    "melt",
    "merge",
    "option_context",
    "options",  # TODO: Move options.py to core, not all objects should be public
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
    "register_dataframe_accessor",
    "register_index_accessor",
    "register_series_accessor",
    "set_option",
    "testing",
    "to_datetime",
    "to_numeric",
    "unique",
    "unstack",
    "utils",
]
