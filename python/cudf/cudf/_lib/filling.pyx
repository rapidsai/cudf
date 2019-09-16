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

from libc.stdint cimport uintptr_t


def repeat(input, repeats):
    from cudf.core.column import Column

    cdef gdf_column* c_input_col = column_view_from_column(input)
    cdef gdf_column* c_repeats_col = column_view_from_column(repeats)
    cdef gdf_column c_result_column

    with nogil:
        c_result_column = cpp_repeat(
            c_input_col[0],
            c_repeats_col[0])

    free_column(c_input_col)
    free_column(c_repeats_col)

    return gdf_column_to_column(&c_result_column)
