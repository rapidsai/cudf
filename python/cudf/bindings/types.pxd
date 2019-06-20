# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *

from libcpp.vector cimport vector


cdef extern from "table.hpp" namespace "cudf" nogil:

    cdef cppclass table:

        table(gdf_column* cols[], gdf_size_type num_cols) except +

        table(const vector[gdf_column*] cols) except +

        table() except +

        gdf_column** begin() except +

        gdf_column** end() except +

        gdf_column* get_column(gdf_index_type index) except +
 
        gdf_size_type num_columns() except +

        gdf_size_type num_rows() except +

# Todo? add const overloads 
#        const gdf_column* const* begin() const except +
#        gdf_column const* const* end() const
#        gdf_column const* get_column(gdf_index_type index) const except +

