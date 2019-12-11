# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *


cdef extern from "cudf/legacy/reshape.hpp" namespace "cudf" nogil:

    cdef gdf_column stack(
        const cudf_table & input
    ) except +
