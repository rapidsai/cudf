# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from typing import Literal

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.device_buffer cimport device_buffer


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
        cdef object _distinct_count

    cdef column_view _view(self, size_type null_count) except *
    cdef column_view view(self) except *
    cdef mutable_column_view mutable_view(self) except *
    cpdef to_pylibcudf(self, mode: Literal["read", "write"])

    @staticmethod
    cdef Column from_unique_ptr(
        unique_ptr[column] c_col, bint data_ptr_exposed=*
    )

    @staticmethod
    cdef Column from_column_view(column_view, object)

    cdef size_type compute_null_count(self) except? 0
