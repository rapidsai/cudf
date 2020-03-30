# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *


cdef extern from "cudf/cudf.h" nogil:

    cdef gdf_error gdf_column_concat(
        gdf_column *output,
        gdf_column *columns_to_concat[],
        int num_columns
    ) except +
