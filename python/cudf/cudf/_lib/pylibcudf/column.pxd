# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from rmm._lib.device_buffer cimport DeviceBuffer

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.types cimport size_type

from .column_view cimport ColumnView


cdef class ColumnContents:
    cdef DeviceBuffer data
    cdef DeviceBuffer null_mask
    cdef list children


cdef class Column:
    cdef cbool released
    cdef unique_ptr[column] c_obj
    cdef column * get(self)
    cpdef size_type size(self)
    cpdef size_type null_count(self)
    cpdef cbool has_nulls(self)
    cpdef ColumnView view(self)
    # cpdef data_type type(self)
    # cpdef column_view view()
    # cpdef mutable_column_view mutable_view()
    cpdef ColumnContents release(self)
    cdef int _raise_if_released(self) except 1


cpdef Column Column_from_ColumnView(ColumnView cv)
