# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *
from libcpp.memory cimport unique_ptr

cdef class Scalar:

    cdef unique_ptr[scalar] c_value

    @staticmethod
    cdef Scalar from_unique_ptr(unique_ptr[scalar] ptr)
