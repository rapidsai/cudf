# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.scalar.scalar cimport scalar


cdef class Scalar:
    cdef unique_ptr[scalar] c_value

    cdef object _host_value
    cdef object _host_dtype

    cdef const scalar* get_raw_ptr(self)

    @staticmethod
    cdef Scalar from_unique_ptr(unique_ptr[scalar] ptr)

    cpdef bool is_valid(Scalar s)
