# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from rmm._lib.device_buffer cimport DeviceBuffer

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.types cimport size_type

# TODO: Don't really like this circular import
from ..types cimport DataType
from .column_view cimport ColumnView


cdef class ColumnContents:
    cdef DeviceBuffer data
    cdef DeviceBuffer null_mask
    cdef list children


cdef class Column:
    cdef cbool released
    cdef unique_ptr[column] c_obj
    cdef column * get(self) noexcept
    cpdef size_type size(self) except -1
    cpdef size_type null_count(self) except -1
    cpdef cbool has_nulls(self) except *
    cpdef DataType type(self)
    cpdef ColumnView view(self)
    cpdef ColumnContents release(self)
    cdef int _raise_if_released(self) except 1

    @staticmethod
    cdef Column from_column(unique_ptr[column] col)


cpdef Column Column_from_ColumnView(ColumnView cv)
