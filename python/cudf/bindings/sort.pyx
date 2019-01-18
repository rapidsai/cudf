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


cpdef apply_order_by(in_cols, out_indices, ascending=True, na_position=1):
    '''
      Call gdf_order_by to retrieve a column of indices of the sorted order
      of rows.
    '''
    cdef gdf_column** input_columns = <gdf_column**>malloc(len(in_cols) * sizeof(gdf_column*))
    for idx, col in enumerate(in_cols):
        check_gdf_compatibility(col)
        input_columns[idx] = column_view_from_column(col)
    
    cdef uintptr_t asc_desc = get_ctype_ptr(ascending)

    cdef size_t num_inputs = len(in_cols)

    check_gdf_compatibility(out_indices)
    cdef gdf_column* output_indices = column_view_from_column(out_indices)

    cdef int flag_nulls_are_smallest = na_position

    cdef gdf_error result 
    
    with nogil:
        result = gdf_order_by(<gdf_column**> input_columns,
                              <int8_t*> asc_desc,
                              <size_t> num_inputs,
                              <gdf_column*> output_indices,
                              <int> flag_nulls_are_smallest)
    
    check_gdf_error(result)
