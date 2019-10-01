# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.
import itertools

import numpy as np
import pandas as pd
import pyarrow as pa

from libc.stdint cimport uintptr_t, int8_t
from libc.stdlib cimport malloc, free
import rmm

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *

cimport cudf._lib.includes.sort as cpp_sort

cpdef order_by(in_cols, out_indices, ascending=True, na_position=1):
    '''
      Call gdf_order_by to retrieve a column of indices of the sorted order
      of rows.
    '''
    cdef gdf_column** input_columns = <gdf_column**>malloc(
        len(in_cols) * sizeof(gdf_column*)
    )
    for idx, col in enumerate(in_cols):
        check_gdf_compatibility(col)
        input_columns[idx] = column_view_from_column(col)

    cdef uintptr_t asc_desc = get_ctype_ptr(ascending)

    cdef size_t num_inputs = len(in_cols)

    check_gdf_compatibility(out_indices)
    cdef gdf_column* output_indices = column_view_from_column(out_indices)

    if na_position == 1:
        null_sort_behavior_api = 'null_as_smallest'
    else:
        null_sort_behavior_api = 'null_as_largest'

    cdef gdf_context* context = create_context_view(
        0,
        'sort',
        0,
        0,
        0,
        null_sort_behavior_api,
        False
    )

    cdef gdf_error result

    with nogil:
        result = cpp_sort.gdf_order_by(
            <gdf_column**> input_columns,
            <int8_t*> asc_desc,
            <size_t> num_inputs,
            <gdf_column*> output_indices,
            <gdf_context*> context
        )

    check_gdf_error(result)
    free(context)


cpdef digitize(column, bins, right=False):
    check_gdf_compatibility(column)
    cdef gdf_column* in_col = column_view_from_column(column)

    check_gdf_compatibility(bins)
    cdef gdf_column* bins_col = column_view_from_column(bins)

    cdef bool cright = right
    cdef gdf_error result
    out = rmm.device_array(len(column), dtype=np.int32)
    cdef uintptr_t out_ptr = get_ctype_ptr(out)

    with nogil:
        result = cpp_sort.gdf_digitize(
            <gdf_column*> in_col,
            <gdf_column*> bins_col,
            <bool> cright,
            <gdf_index_type*> out_ptr
        )

    check_gdf_error(result)
    return out
