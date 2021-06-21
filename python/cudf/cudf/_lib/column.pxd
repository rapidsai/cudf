# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from rmm._lib.device_buffer cimport device_buffer

from cudf._lib.cpp.types cimport size_type

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport (
    column_view, mutable_column_view
)


cdef class Column:
    cdef public:
        cdef int _offset
        cdef int _size
        cdef object _dtype
        cdef object _base_children
        cdef object _base_data
        cdef object _base_mask
        cdef object _children
        cdef object _data
        cdef object _mask
        cdef object _null_count

    cdef column_view _view(self, size_type null_count) except *
    cdef column_view view(self) except *
    cdef mutable_column_view mutable_view(self) except *

    @staticmethod
    cdef Column from_unique_ptr(unique_ptr[column] c_col)

    @staticmethod
    cdef Column from_column_view(column_view, object)

    cdef size_type compute_null_count(self) except? 0
