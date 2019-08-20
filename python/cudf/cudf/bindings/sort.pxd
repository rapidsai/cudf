# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf/cudf.h" nogil:

    ctypedef enum order_by_type:
        GDF_ORDER_ASC,
        GDF_ORDER_DESC

    cdef gdf_error gdf_order_by(
        gdf_column** input_columns,
        int8_t* asc_desc,
        size_t num_inputs,
        gdf_column* output_indices,
        gdf_context* ctxt
    ) except +
