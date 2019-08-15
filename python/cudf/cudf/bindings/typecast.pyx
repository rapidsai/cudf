# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.unaryops cimport *

import numpy as np
import pandas as pd


def apply_cast(incol, dtype=np.float64):
    """
    Return a Column with values in `incol` casted to `dtype`.
    Currently supports numeric and datetime dtypes.
    """
    check_gdf_compatibility(incol)
    dtype = np.dtype(np.float64 if dtype is None else dtype)

    if pd.api.types.is_dtype_equal(incol.dtype, dtype):
        return incol

    cdef gdf_column* c_incol = column_view_from_column(incol)
    cdef gdf_dtype c_out_dtype = gdf_dtype_from_value(incol, dtype)
    cdef uintptr_t c_category
    cdef gdf_dtype_extra_info c_out_info = gdf_dtype_extra_info(
        time_unit=np_dtype_to_gdf_time_unit(dtype),
        category=<void*>c_category
    )

    cdef gdf_column c_out_col

    with nogil:
        c_out_col = cast(
            c_incol[0],
            c_out_dtype,
            c_out_info
        )

    free_column(c_incol)

    return gdf_column_to_column(&c_out_col)
