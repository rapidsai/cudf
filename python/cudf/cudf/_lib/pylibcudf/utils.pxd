# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport bitmask_type


cdef void * int_to_void_ptr(ptr)
cdef bitmask_type * int_to_bitmask_ptr(ptr)
