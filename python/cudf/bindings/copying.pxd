# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.types cimport table as cudf_table


cdef extern from "copying.hpp" namespace "cudf" nogil:
    
    cdef void scatter(const cudf_table* source_table, const gdf_index_type* scatter_map,
                      cudf_table* destination_table) except +

    cdef void gather(const cudf_table * source_table, const gdf_index_type* gather_map,
                     cudf_table* destination_table) except +

cdef gdf_column** cols_view_from_cols(cols)
cdef free_table(cudf_table* table, gdf_column** cols)


