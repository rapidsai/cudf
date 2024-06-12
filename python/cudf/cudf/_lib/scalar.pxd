# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from rmm._lib.memory_resource cimport DeviceMemoryResource

from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar


cdef class DeviceScalar:
    cdef public object c_value

    cdef object _dtype

    cdef const scalar* get_raw_ptr(self) except *

    @staticmethod
    cdef DeviceScalar from_unique_ptr(unique_ptr[scalar] ptr, dtype=*)

    @staticmethod
    cdef DeviceScalar from_pylibcudf(pscalar, dtype=*)

    cdef void _set_dtype(self, dtype=*)

    cpdef bool is_valid(DeviceScalar s)
