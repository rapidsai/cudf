# Copyright (c) 2023, NVIDIA CORPORATION.

from libc.stdint cimport uintptr_t

from cudf._lib.cpp.types cimport bitmask_type


# Helpers
cdef void * int_to_void_ptr(ptr):
    # Cython will not cast a Python integer directly to a pointer, so the
    # intermediate cast to a uintptr_t is necessary
    return <void*><uintptr_t>(ptr)


cdef bitmask_type * int_to_bitmask_ptr(ptr):
    # Cython will not cast a Python integer directly to a pointer, so the
    # intermediate cast to a uintptr_t is necessary
    return <bitmask_type*><uintptr_t>(ptr)
