# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.scalar.scalar cimport scalar


cdef class DeviceScalar:
    cdef unique_ptr[scalar] c_value

    cdef const scalar* get_raw_ptr(self) except *

    @staticmethod
    cdef DeviceScalar from_unique_ptr(unique_ptr[scalar] ptr)

    cpdef bool is_valid(DeviceScalar s)
