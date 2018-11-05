# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from .cudf_cpp cimport *
from .cudf_cpp import *

import numpy as np
import pandas as pd
import pyarrow as pa

cimport numpy as np

from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm


from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free


cpdef apply_sort(col_keys, col_vals, ascending=True):
    """Inplace sort
    """
    nelem = len(col_keys)
    begin_bit = 0
    end_bit = col_keys.dtype.itemsize * 8
    cdef gdf_radixsort_plan_type* plan
    cdef bool descending = not ascending
    plan = gdf_radixsort_plan(<size_t>nelem,
                              <bool>descending,
                              <unsigned>begin_bit,
                              <unsigned>end_bit)
    sizeof_key = col_keys.dtype.itemsize
    sizeof_val = col_vals.dtype.itemsize
    cdef gdf_column* c_col_keys = column_view_from_column(col_keys)
    cdef gdf_column* c_col_vals = column_view_from_column(col_vals)
    cdef gdf_error result
    try:
        gdf_radixsort_plan_setup(plan, sizeof_key, sizeof_val)
        result = gdf_radixsort_generic(plan,
                                    c_col_keys,
                                     c_col_vals)
        check_gdf_error(result)
    finally:
        gdf_radixsort_plan_free(plan)
        free(c_col_keys)
        free(c_col_vals)
