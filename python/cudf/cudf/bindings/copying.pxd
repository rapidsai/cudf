# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf/copying.hpp" namespace "cudf" nogil:

    cdef void gather(
        const cudf_table * source_table,
        const gdf_index_type* gather_map,
        cudf_table* destination_table
    ) except +

    cdef gdf_column copy(
        const gdf_column &input
    ) except +

    cudf_table scatter(
        const cudf_table source,
        const gdf_index_type* scatter_map,
        const cudf_table target)

    cdef void copy_range(
        gdf_column *out_column,
        const gdf_column in_column,
        gdf_index_type out_begin,
        gdf_index_type out_end,
        gdf_index_type in_begin)
