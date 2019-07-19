# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *

cdef extern from "rolling.hpp" namespace "cudf" nogil:
    gdf_column* rolling_window(
        const gdf_column &input_col,
        gdf_size_type window,
        gdf_size_type min_periods,
        gdf_size_type forward_window,
        gdf_agg_op agg_type,
        const gdf_size_type *window_col,
        const gdf_size_type *min_periods_col,
        const gdf_size_type *forward_window_col
    ) except +
