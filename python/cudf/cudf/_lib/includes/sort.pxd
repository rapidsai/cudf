# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *


cdef extern from "cudf/cudf.h" nogil:

    cdef gdf_error gdf_order_by(
        gdf_column** input_columns,
        int8_t* asc_desc,
        size_t num_inputs,
        gdf_column* output_indices,
        gdf_context* ctxt
    ) except +
