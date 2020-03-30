# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *

cdef extern from "cudf/legacy/replace.hpp" namespace "cudf" nogil:

    cdef gdf_column replace_nulls(
        const gdf_column& inp,
        const gdf_column& replacement_values
    ) except +

    cdef gdf_column replace_nulls(
        const gdf_column& inp,
        const gdf_scalar& replacement_value
    ) except +

    cdef gdf_column find_and_replace_all(
        const gdf_column &input_col,
        const gdf_column &values_to_replace,
        const gdf_column &replacement_values
    ) except +
