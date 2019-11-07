# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._libxx.column cimport Column
from libc.stdlib cimport free
from libcpp.vector cimport vector

from cudf._lib.includes.concat cimport gdf_column_concat


def _column_concat(cols_to_concat, Column output_col):
    cdef gdf_column* c_output_col = column_view_from_column(output_col)
    cdef vector[gdf_column*] c_input_cols
    cdef int num_cols = len(cols_to_concat)

    cdef Column col
    for col in cols_to_concat:
        c_input_cols.push_back(column_view_from_column(col))

    with nogil:
        result = gdf_column_concat(
            c_output_col,
            c_input_cols.data(),
            num_cols
        )

    check_gdf_error(result)

    free_column(c_output_col)
    for c_col in c_input_cols:
        free_column(c_col)

    return output_col
