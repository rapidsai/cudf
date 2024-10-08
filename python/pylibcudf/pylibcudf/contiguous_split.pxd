# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.contiguous_split cimport packed_columns

from .table cimport Table


cdef class PackedColumns:
    cdef unique_ptr[packed_columns] c_obj

    @staticmethod
    cdef PackedColumns from_libcudf(unique_ptr[packed_columns] data)

cpdef PackedColumns pack(Table input)

cpdef Table unpack(PackedColumns input)
