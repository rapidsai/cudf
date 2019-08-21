# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf/join.hpp" nogil:

    cdef cudf_table inner_join(
        const cudf_table left_cols,
        const cudf_table right_cols,
        const cudf_table left_on_ind,
        const cudf_table right_on_ind,
        const vector[pair[int, int]] joining_cols,
        const cudf_table * out_ind,
        gdf_context * join_context
    ) except +

    cdef cudf_table left_join(
        const cudf_table left_cols,
        const cudf_table right_cols,
        const cudf_table left_on_ind,
        const cudf_table right_on_ind,
        const vector[pair[int, int]] joining_cols,
        const cudf_table * out_ind,
        gdf_context * join_context
    ) except +

    cdef cudf_table full_join(
        const cudf_table left_cols,
        const cudf_table right_cols,
        const cudf_table left_on_ind,
        const cudf_table right_on_ind,
        const vector[pair[int, int]] joining_cols,
        const cudf_table * out_ind,
        gdf_context * join_context
    ) except +
