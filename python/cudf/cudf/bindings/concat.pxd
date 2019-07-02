# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf.h" nogil:

    cdef gdf_error gdf_column_concat(
        gdf_column *output,
        gdf_column *columns_to_concat[],
        int num_columns
    ) except +

