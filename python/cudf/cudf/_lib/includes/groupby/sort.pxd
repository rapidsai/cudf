# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.utility cimport pair

from cudf._lib.cudf import *
from cudf._lib.cudf cimport *

cdef extern from "cudf/legacy/groupby.hpp" nogil:
    cdef pair[cudf_table, gdf_column] gdf_group_by_without_aggregations(
        const cudf_table  cols,
        size_type num_key_cols,
        const size_type* key_col_indices,
        gdf_context* context
    ) except +
