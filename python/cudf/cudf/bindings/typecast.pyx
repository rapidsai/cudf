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


def apply_cast(incol, dtype=np.float64):
    """
      Cast from incol.dtype to outcol.dtype
    """

    check_gdf_compatibility(incol)

    cdef gdf_column* c_incol = column_view_from_column(incol)

    dtype = np.dtype(np.float64 if dtype is None else dtype)
    cdef gdf_dtype c_out_dtype = gdf_dtype_from_value(incol, dtype)
    cdef uintptr_t category

    cdef gdf_dtype_extra_info c_out_info = gdf_dtype_extra_info(
        time_unit=np_dtype_to_gdf_time_unit(dtype),
        category=<void*>category
    )

    cdef gdf_column result

    with nogil:
        result = cast(
            c_incol[0],
            c_out_dtype,
            c_out_info
        )

    free(c_incol)
    data, mask = gdf_column_to_column_mem(&result)
    return Column.from_mem_views(data, mask)
