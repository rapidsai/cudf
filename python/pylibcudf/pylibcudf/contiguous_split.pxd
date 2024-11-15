# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.contiguous_split cimport packed_columns

from .gpumemoryview cimport gpumemoryview
from .table cimport Table


cdef class PackedColumns:
    cdef unique_ptr[packed_columns] c_obj

    @staticmethod
    cdef PackedColumns from_libcudf(unique_ptr[packed_columns] data)
    cpdef tuple release(self)

cpdef PackedColumns pack(Table input)

cpdef Table unpack(PackedColumns input)

cpdef Table unpack_from_memoryviews(memoryview metadata, gpumemoryview gpu_data)
