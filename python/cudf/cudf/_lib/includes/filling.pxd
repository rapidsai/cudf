# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *


cdef extern from "cudf/filling.hpp" namespace "cudf" nogil:

    cdef void fill(
        gdf_column * column,
        const gdf_scalar & value,
        gdf_index_type begin,
        gdf_index_type end
    ) except +

    cdef gdf_column repeat(
        const gdf_column & input,
        const gdf_column & count
    ) except +

    cdef gdf_column repeat(
        const gdf_column & input,
        const gdf_scalar & count
    ) except +
