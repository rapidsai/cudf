# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.stream_compaction import *

from cudf.dataframe.column import Column


def cpp_apply_boolean_mask(inp, mask):
    cdef gdf_column *inp_col = column_view_from_column(inp)
    cdef gdf_column *mask_col = column_view_from_column(mask)
    cdef gdf_column result  = apply_boolean_mask(inp_col[0], mask_col[0])
    data, mask = gdf_column_to_column_mem(&result)
    return Column.from_mem_views(data, mask)


def cpp_drop_nulls(inp):
    cdef gdf_column *inp_col = column_view_from_column(inp)
    cdef gdf_column result = drop_nulls(inp_col[0])
    data, mask = gdf_column_to_column_mem(&result)
    return Column.from_mem_views(data, mask)
