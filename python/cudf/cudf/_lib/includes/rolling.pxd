# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *

from libcpp.string cimport string

cdef extern from "cudf/legacy/rolling.hpp" namespace "cudf" nogil:
    gdf_column* rolling_window(
        const gdf_column &input_col,
        size_type window,
        size_type min_periods,
        size_type forward_window,
        gdf_agg_op agg_type,
        const size_type *window_col,
        const size_type *min_periods_col,
        const size_type *forward_window_col
    ) except +


cdef extern from "cudf/legacy/rolling.hpp" namespace "cudf" nogil:
    gdf_column rolling_window(
        const gdf_column &input_col,
        size_type window,
        size_type min_periods,
        size_type forward_window,
        const string& user_defined_aggregator,
        gdf_agg_op agg_op,
        gdf_dtype output_type,
        const size_type *window_col,
        const size_type *min_periods_col,
        const size_type *forward_window_col
    ) except +
