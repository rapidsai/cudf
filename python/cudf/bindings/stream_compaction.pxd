# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *

cdef extern from "stream_compaction.hpp" namespace "cudf" nogil:

    cdef gdf_column apply_boolean_mask(const gdf_column &input,
                                       const gdf_column &boolean_mask) except +

    cdef gdf_column drop_nulls(const gdf_column &input) except +
