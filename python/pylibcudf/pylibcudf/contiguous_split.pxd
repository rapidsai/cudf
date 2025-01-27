# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.libcudf.contiguous_split cimport packed_columns

from .gpumemoryview cimport gpumemoryview
from .table cimport Table


cdef class HostBuffer:
    cdef unique_ptr[vector[uint8_t]] c_obj
    cdef size_t nbytes
    cdef Py_ssize_t[1] shape
    cdef Py_ssize_t[1] strides

    @staticmethod
    cdef HostBuffer from_unique_ptr(
        unique_ptr[vector[uint8_t]] vec
    )

cdef class PackedColumns:
    cdef unique_ptr[packed_columns] c_obj

    @staticmethod
    cdef PackedColumns from_libcudf(unique_ptr[packed_columns] data)
    cpdef tuple release(self)

cpdef PackedColumns pack(Table input)

cpdef Table unpack(PackedColumns input)

cpdef Table unpack_from_memoryviews(memoryview metadata, gpumemoryview gpu_data)
