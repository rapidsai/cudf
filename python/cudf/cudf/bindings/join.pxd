# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf.h" nogil:

    cdef gdf_error gdf_inner_join(
        gdf_column **left_cols,
        int num_left_cols,
        int left_join_cols[],
        gdf_column **right_cols,
        int num_right_cols,
        int right_join_cols[],
        int num_cols_to_join,
        int result_num_cols,
        gdf_column **result_cols,
        gdf_column * left_indices,
        gdf_column * right_indices,
        gdf_context *join_context
    ) except +

    cdef gdf_error gdf_left_join(
        gdf_column **left_cols,
        int num_left_cols,
        int left_join_cols[],
        gdf_column **right_cols,
        int num_right_cols,
        int right_join_cols[],
        int num_cols_to_join,
        int result_num_cols,
        gdf_column **result_cols,
        gdf_column * left_indices,
        gdf_column * right_indices,
        gdf_context *join_context
    ) except +

    cdef gdf_error gdf_full_join(
        gdf_column **left_cols,
        int num_left_cols,
        int left_join_cols[],
        gdf_column **right_cols,
        int num_right_cols,
        int right_join_cols[],
        int num_cols_to_join,
        int result_num_cols,
        gdf_column **result_cols,
        gdf_column * left_indices,
        gdf_column * right_indices,
        gdf_context *join_context
    ) except +
