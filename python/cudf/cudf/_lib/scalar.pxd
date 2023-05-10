# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from rmm._lib.memory_resource cimport DeviceMemoryResource

from cudf._lib.cpp.scalar.scalar cimport scalar


cdef class DeviceScalar:
    cdef unique_ptr[scalar] c_value

    # Holds a reference to the DeviceMemoryResource used for allocation.
    # Ensures the MR does not get destroyed before this DeviceBuffer. `mr` is
    # needed for deallocation
    cdef DeviceMemoryResource mr

    cdef object _dtype

    cdef const scalar* get_raw_ptr(self) except *

    @staticmethod
    cdef DeviceScalar from_unique_ptr(unique_ptr[scalar] ptr, dtype=*)

    cpdef bool is_valid(DeviceScalar s)
