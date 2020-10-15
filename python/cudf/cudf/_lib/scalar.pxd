# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.scalar.scalar cimport scalar


cdef class Scalar:
    cdef _ScalarUptrWrapper uptr

    cdef object _host_value
    cdef object _host_dtype

    cdef _ScalarUptrWrapper get_uptr(self)

    @staticmethod
    cdef Scalar from_unique_ptr(unique_ptr[scalar] ptr)

    cpdef bool is_valid(Scalar s)

cdef class _ScalarUptrWrapper:
    cdef unique_ptr[scalar] _device_uptr
