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
)
from cudf._lib.includes.reshape cimport (
    stack as cpp_stack,
)


def stack(input):
    from cudf.core.column import as_column

    cdef cudf_table* c_input_table = table_from_columns(input)
    cdef gdf_column c_result_column

    with nogil:
        c_result_column = cpp_stack(
            c_input_table[0])

    del c_input_table

    return gdf_column_to_column(&c_result_column)
