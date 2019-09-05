# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *


cdef extern from "cudf/shifting.hpp" namespace "cudf" nogil:

    cdef cudf_table shift(
        const cudf_table& in_table,
        gdf_index_type offset,
        const gdf_scalar& fill_value
     ) except +

    cdef gdf_column shift(
        const gdf_column& in_column,
        gdf_index_type offset,
        const gdf_scalar& fill_value
    ) except +
