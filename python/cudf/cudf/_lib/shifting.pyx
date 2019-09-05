# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf.utils.dtypes import is_scalar

from cudf._lib.utils cimport table_from_dataframe
from cudf._lib.includes.shifting cimport (
    shift as cpp_shift
)

def shift_column(input_column, period, fill_value):
    """
        Call cudf::shift
    """
    cdef gdf_column* c_input_column = column_view_from_column(input_column)
    cdef gdf_scalar* c_fill_value = gdf_scalar_from_scalar(fill_value)
    cdef gdf_column c_output_column
    cdef gdf_index_type c_period = period

    with nogil:
        c_output_column = cpp_shift(
            c_input_column[0],
            c_period,
            c_fill_value[0]
        )

    free_column(c_input_column)

    return gdf_column_to_column(&c_output_column)
