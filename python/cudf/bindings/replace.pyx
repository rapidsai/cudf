# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.replace cimport *
from cudf.dataframe.column import Column
from cudf.utils.utils import is_single_value

from libc.stdlib cimport calloc, malloc, free


cpdef replace(input_col, values_to_replace, replacement_values):
    """
        Call cudf::find_and_replace_all
    """
    cdef gdf_column* c_input_col = \
                            column_view_from_column(input_col)
    cdef gdf_column* c_values_to_replace = \
                            column_view_from_column(values_to_replace)
    cdef gdf_column* c_replacement_values = \
                            column_view_from_column(replacement_values)

    cdef gdf_column* output = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        output[0] = find_and_replace_all(c_input_col[0],
                                      c_values_to_replace[0],
                                      c_replacement_values[0])

    data, mask = gdf_column_to_column_mem(output)

    free(c_values_to_replace)
    free(c_replacement_values)
    free(c_input_col)
    free(output)

    return Column.from_mem_views(data, mask)



cdef apply_replace_nulls_column(inp, replacement):
    cdef gdf_column* inp_col = column_view_from_column(inp)
    cdef gdf_column* replacement_col = column_view_from_column(replacement)
    cdef gdf_column result

    with nogil:
        result = replace_nulls(inp_col[0], replacement_col[0])

    data, mask = gdf_column_to_column_mem(&result)
    return Column.from_mem_views(data, mask)

cdef apply_replace_nulls_scalar(inp, replacement):
    cdef gdf_column* inp_col = column_view_from_column(inp)
    cdef gdf_scalar* replacement_scalar = gdf_scalar_from_scalar(replacement)
    cdef gdf_column result

    with nogil:
        result = replace_nulls(inp_col[0], replacement_scalar[0])

    data, mask = gdf_column_to_column_mem(&result)
    return Column.from_mem_views(data, mask)


cpdef apply_replace_nulls(inp, replacement):
    """
    Call replace_nulls
    """

    if is_single_value(replacement):
        return apply_replace_nulls_scalar(inp, replacement)
    else:
        return apply_replace_nulls_column(inp, replacement)
