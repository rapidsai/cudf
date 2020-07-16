# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp cimport bool
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view


ctypedef bool underlying_type_t_order
ctypedef bool underlying_type_t_null_order
ctypedef bool underlying_type_t_sorted
ctypedef int32_t underlying_type_t_interpolation
ctypedef int32_t underlying_type_t_type_id
ctypedef bool underlying_type_t_null_policy

cdef dtype_from_column_view(column_view cv)
