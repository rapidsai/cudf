# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.libcudf.contiguous_split cimport packed_columns, chunked_pack
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

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

cdef class ChunkedPack:
    cdef unique_ptr[chunked_pack] c_obj
    cdef Table table
    cdef DeviceMemoryResource mr
    cdef Stream stream

    cpdef bool has_next(self)
    cpdef size_t next(self, DeviceBuffer buf)
    cpdef size_t get_total_contiguous_size(self)
    cpdef memoryview build_metadata(self)
    cpdef tuple pack_to_host(self, DeviceBuffer buf)


cpdef PackedColumns pack(Table input)

cpdef Table unpack(PackedColumns input)

cpdef Table unpack_from_memoryviews(memoryview metadata, gpumemoryview gpu_data)
