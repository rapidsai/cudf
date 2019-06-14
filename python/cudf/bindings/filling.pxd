# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from cudf.bindings.cudf_cpp cimport *

cdef extern from "filling.hpp" namespace "cudf" nogil:
    cdef void fill(gdf_column *column, const gdf_scalar value,
                   gdf_index_type begin, gdf_index_type end) except +
