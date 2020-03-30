# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *


cdef extern from "cudf/cudf.h" nogil:

    cdef gdf_error gdf_transpose(
        size_type ncols,
        gdf_column** in_cols,
        gdf_column** out_cols
    ) except +
