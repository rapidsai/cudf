# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "copying.hpp" namespace "cudf" nogil:
    
    cdef void scatter(const cudf_table* source_table, const gdf_index_type* scatter_map,
                      cudf_table* destination_table) except +

    cdef void gather(const cudf_table * source_table, const gdf_index_type* gather_map,
                     cudf_table* destination_table) except +
