# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *
from libcpp.memory cimport unique_ptr
cdef class Scalar:
    cdef unique_ptr[scalar] c_value

    @staticmethod
    cdef Scalar from_unique_ptr(unique_ptr[scalar] ptr)
