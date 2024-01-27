# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from rmm._lib.memory_resource cimport DeviceMemoryResource

# TODO: Would like to remove this cimport, but it will require some more work
# to excise all C code in scalar.pyx that relies on using the C API of the
# pylibcudf Scalar underlying the DeviceScalar.
from cudf._lib cimport pylibcudf
from cudf._lib.cpp.scalar.scalar cimport scalar


cdef class DeviceScalar:
    cdef pylibcudf.Scalar c_value

    cdef object _dtype

    cdef const scalar* get_raw_ptr(self) except *

    @staticmethod
    cdef DeviceScalar from_unique_ptr(unique_ptr[scalar] ptr, dtype=*)

    @staticmethod
    cdef DeviceScalar from_pylibcudf(pylibcudf.Scalar scalar, dtype=*)

    cdef void _set_dtype(self, dtype=*)

    cpdef bool is_valid(DeviceScalar s)
