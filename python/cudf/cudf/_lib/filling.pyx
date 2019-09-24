# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *

import cudf.utils.utils as utils
from cudf._lib.utils cimport (
    columns_from_table,
    table_from_columns,
    table_to_dataframe
)
from cudf._lib.includes.filling cimport (
    fill as cpp_fill,
    repeat as cpp_repeat,
)

import numpy as np

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free


def repeat(input, repeats):
    from cudf.core.column import as_column

    cdef gdf_scalar* c_repeats_scalar = NULL
    cdef gdf_column* c_repeats_col = NULL

    cdef cudf_table* c_input_table = table_from_columns(input)
    cdef cudf_table c_result_table

    if np.isscalar(repeats):
        repeats = np.dtype("int32").type(repeats)
        c_repeats_scalar = gdf_scalar_from_scalar(repeats)
        with nogil:
            c_result_table = cpp_repeat(
                c_input_table[0],
                c_repeats_scalar[0])
    else:
        repeats = as_column(repeats).astype("int32")
        c_repeats_col = column_view_from_column(repeats)
        with nogil:
            c_result_table = cpp_repeat(
                c_input_table[0],
                c_repeats_col[0])

    free(c_repeats_scalar)
    del c_input_table
    free_column(c_repeats_col)

    return columns_from_table(&c_result_table)
