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

from libc.stdlib cimport calloc, malloc, free



cpdef replace(col, old_values, new_values):
    """
        Call gdf_find_and_replace_all
    """
    cdef gdf_column* c_col = column_view_from_column(col)
    cdef gdf_column* c_old_values = column_view_from_column(old_values)
    cdef gdf_column* c_new_values = column_view_from_column(new_values)

    gdf_find_and_replace_all(c_col, c_old_values, c_new_values)


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
    from cudf.indexing import _is_single_value

    if _is_single_value(replacement):
        return apply_replace_nulls_scalar(inp, replacement)
    else:
        return apply_replace_nulls_column(inp, replacement)
