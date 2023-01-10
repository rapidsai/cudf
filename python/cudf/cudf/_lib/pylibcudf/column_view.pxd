# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type


cdef class ColumnView:
    cdef unique_ptr[column_view] c_obj
    cdef column_view * get(self) nogil
    cpdef size_type size(self)
    cpdef size_type null_count(self)

    @staticmethod
    cdef from_column_view(column_view cv)
