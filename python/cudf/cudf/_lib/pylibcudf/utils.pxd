# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport bitmask_type


cdef void * int_to_void_ptr(Py_ssize_t ptr) nogil
cdef bitmask_type * int_to_bitmask_ptr(Py_ssize_t ptr) nogil
