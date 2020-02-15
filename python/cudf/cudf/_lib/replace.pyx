# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from libc.stdlib cimport free

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._libxx.column cimport Column
from cudf.utils.dtypes import is_scalar

cimport cudf._lib.includes.replace as cpp_replace


cpdef replace(Column input_col, Column values_to_replace,
              Column replacement_values):
    """
        Call cudf::find_and_replace_all
    """
    cdef gdf_column* c_input_col = column_view_from_column(input_col)
    cdef gdf_column* c_values_to_replace = column_view_from_column(
        values_to_replace
    )
    cdef gdf_column* c_replacement_values = column_view_from_column(
        replacement_values
    )

    cdef gdf_column c_out_col

    with nogil:
        c_out_col = cpp_replace.find_and_replace_all(
            c_input_col[0],
            c_values_to_replace[0],
            c_replacement_values[0]
        )

    free_column(c_values_to_replace)
    free_column(c_replacement_values)
    free_column(c_input_col)

    return gdf_column_to_column(&c_out_col)


cdef replace_nulls_column(Column inp, Column replacement):
    cdef gdf_column* c_input_col = column_view_from_column(inp)
    cdef gdf_column* replacement_col = column_view_from_column(replacement)
    cdef gdf_column c_out_col

    with nogil:
        c_out_col = cpp_replace.replace_nulls(
            c_input_col[0],
            replacement_col[0]
        )

    free_column(replacement_col)
    free_column(c_input_col)

    return gdf_column_to_column(&c_out_col)

cdef replace_nulls_scalar(Column inp, replacement):
    cdef gdf_column* c_input_col = column_view_from_column(inp)
    cdef gdf_scalar* replacement_scalar = gdf_scalar_from_scalar(replacement)
    cdef gdf_column c_out_col

    with nogil:
        c_out_col = cpp_replace.replace_nulls(
            c_input_col[0],
            replacement_scalar[0]
        )

    free(replacement_scalar)
    free_column(c_input_col)

    return gdf_column_to_column(&c_out_col)


cpdef replace_nulls(inp, replacement):
    """
    Call replace_nulls
    """

    if is_scalar(replacement):
        return replace_nulls_scalar(inp, replacement)
    else:
        return replace_nulls_column(inp, replacement)
