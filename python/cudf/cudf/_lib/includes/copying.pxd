# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from libcpp.vector cimport vector
from libcpp.pair cimport pair


cdef extern from "cudf/copying.hpp" namespace "cudf" nogil:

    cdef cudf_table gather(
        const cudf_table * source_table,
        const gdf_column gather_map,
        bool bounds_check
    ) except +

    cdef gdf_column copy(
        const gdf_column &input
    ) except +

    cdef cudf_table scatter(
        const cudf_table source,
        const gdf_column scatter_map,
        const cudf_table target,
        bool bounds_check
    ) except +

    cdef void copy_range(
        gdf_column *out_column,
        const gdf_column in_column,
        gdf_index_type out_begin,
        gdf_index_type out_end,
        gdf_index_type in_begin
    ) except +

    cdef vector[cudf_table] scatter_to_tables(
        const cudf_table& input,
        const gdf_column& scatter_map
    ) except +
