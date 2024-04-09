# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from rmm._lib.memory_resource cimport DeviceMemoryResource

from cudf._lib.cpp.scalar.scalar cimport scalar

from .types cimport DataType


cdef class Scalar:
    cdef unique_ptr[scalar] c_obj
    cdef DataType _data_type

    # Holds a reference to the DeviceMemoryResource used for allocation.
    # Ensures the MR does not get destroyed before this DeviceBuffer. `mr` is
    # needed for deallocation
    cdef DeviceMemoryResource mr

    cdef const scalar* get(self) noexcept nogil

    cpdef DataType type(self)
    cpdef bool is_valid(self)

    @staticmethod
    cdef Scalar from_libcudf(unique_ptr[scalar] libcudf_scalar, dtype=*)
