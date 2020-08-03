# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp cimport bool
from cudf._lib.cpp.types cimport data_type

ctypedef bool underlying_type_t_order
ctypedef bool underlying_type_t_null_order
ctypedef bool underlying_type_t_sorted
ctypedef int32_t underlying_type_t_interpolation
ctypedef int32_t underlying_type_t_type_id
ctypedef bool underlying_type_t_null_policy

cdef class _Dtype:
    cdef data_type get_libcudf_type(self) except *
