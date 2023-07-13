# Copyright (c) 2023, NVIDIA CORPORATION.

from libc.stdint cimport uintptr_t

from cudf._lib.cpp.types cimport bitmask_type


cdef void * int_to_void_ptr(Py_ssize_t ptr) nogil:
    return <void*><uintptr_t>(ptr)


cdef bitmask_type * int_to_bitmask_ptr(Py_ssize_t ptr) nogil:
    return <bitmask_type*><uintptr_t>(ptr)
