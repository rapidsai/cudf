# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf/join.hpp" nogil:

    cdef pair[cudf_table, cudf_table] gdf_inner_join(
        const cudf_table left_cols,
        const vector [int] left_join_cols,
        const cudf_table right_cols,
        const vector [int] right_join_cols,
        gdf_column * left_indices,
        gdf_column * right_indices,
        gdf_context *join_context,
        const vector [int] left_join_result_cols,
        const vector [int] right_join_result_cols
    ) except +

    cdef pair[cudf_table, cudf_table] gdf_left_join(
        const cudf_table left_cols,
        const vector [int] left_join_cols,
        const cudf_table right_cols,
        const vector [int] right_join_cols,
        gdf_column * left_indices,
        gdf_column * right_indices,
        gdf_context *join_context,
        const vector [int] left_join_result_cols,
        const vector [int] right_join_result_cols
    ) except +

    cdef pair[cudf_table, cudf_table] gdf_full_join(
        const cudf_table left_cols,
        const vector [int] left_join_cols,
        const cudf_table right_cols,
        const vector [int] right_join_cols,
        gdf_column * left_indices,
        gdf_column * right_indices,
        gdf_context *join_context,
        const vector [int] left_join_result_cols,
        const vector [int] right_join_result_cols
    ) except +
