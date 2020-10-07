# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp cimport bool

from cudf._lib.cpp.scalar.scalar cimport scalar


cdef class Scalar:
    cdef unique_ptr[scalar] c_value

    @staticmethod
    cdef Scalar from_unique_ptr(unique_ptr[scalar] ptr)

    cpdef bool is_valid(Scalar s)
