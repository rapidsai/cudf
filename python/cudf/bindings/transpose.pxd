# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf.h" nogil:

    cdef gdf_error gdf_transpose(
        gdf_size_type ncols,
        gdf_column** in_cols,
        gdf_column** out_cols) except +
