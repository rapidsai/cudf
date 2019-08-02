# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *
from .cudf_cpp import *
from cudf.bindings.unaryops cimport *
from cudf.dataframe.column import Column
from libc.stdlib cimport free

import numpy as np
import pandas as pd

_time_unit = {
    None: TIME_UNIT_NONE,
    's': TIME_UNIT_s,
    'ms': TIME_UNIT_ms,
    'us': TIME_UNIT_us,
    'ns': TIME_UNIT_ns,
}


def apply_cast(incol, dtype="float64", time_unit=None):
    """
    Return a Column with values in `incol` casted to `dtype`.
    Currently supports numeric and datetime dtypes.
    """

    check_gdf_compatibility(incol)
    dtype = pd.api.types.pandas_dtype(dtype).type

    cdef gdf_column* c_incol = column_view_from_column(incol)

    cdef gdf_dtype c_dtype = dtypes[dtype]
    cdef uintptr_t c_category

    cdef gdf_dtype_extra_info info = gdf_dtype_extra_info(
        time_unit=TIME_UNIT_NONE,
        category=<void*>c_category
    )
    info.time_unit = _time_unit[time_unit]

    cdef gdf_column result

    with nogil:
        result = cast(
            c_incol[0],
            c_dtype,
            info
        )

    free(c_incol)
    data, mask = gdf_column_to_column_mem(&result)
    return Column.from_mem_views(data, mask)
