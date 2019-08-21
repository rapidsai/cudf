# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *


cdef extern from "cudf/cudf.h" nogil:

    cdef gdf_error gdf_order_by(
        gdf_column** input_columns,
        int8_t* asc_desc,
        size_t num_inputs,
        gdf_column* output_indices,
        gdf_context* ctxt
    ) except +
