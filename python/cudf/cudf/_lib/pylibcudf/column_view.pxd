# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column_view cimport column_view


cdef class ColumnView:
    cdef unique_ptr[column_view] c_obj
    cdef column_view get(self) nogil
