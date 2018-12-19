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


from libc.stdint cimport uintptr_t, int8_t
from libc.stdlib cimport calloc, malloc, free


cpdef apply_sort(col_keys, col_vals, ascending=True):
    '''
      Call gdf join for full outer, inner and left joins.
    '''
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

cpdef apply_order_by(in_cols, out_indices, ascending=True, na_position=1):
    '''
      Call gdf_order_by to retrieve a column of indices of the sorted order
      of rows.
    '''
    cdef gdf_column** input_columns = <gdf_column**>malloc(len(in_cols) * sizeof(gdf_column*))
    for col in in_cols:
        input_columns.append(column_view_from_column(col._column))
    
    cdef int8_t* asc_desc = ascending

    <size_t>num_inputs = len(in_cols)

    cdef gdf_column* output_indices = column_view_from_column(out_indices._column)

    flag_nulls_are_smallest = na_position

    cdef gdf_error result =  gdf_order_by(<gdf_column**> input_columns,
                                          <int8_t*> asc_desc,
                                          <size_t> num_inputs,
                                          <gdf_column*> output_indices,
                                          <int> flag_nulls_are_smallest)
    
    check_gdf_error(result)
