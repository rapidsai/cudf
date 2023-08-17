# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport bitmask_type, size_type

from .gpumemoryview cimport gpumemoryview
from .types cimport DataType


cdef class Column:
    # TODO: Should we document these attributes? Should we mark them readonly?
    cdef:
        # Core data
        DataType data_type
        size_type size
        gpumemoryview data
        gpumemoryview mask
        size_type null_count
        size_type offset
        # children: List[Column]
        list children
        size_type _num_children

    cdef column_view view(self) nogil

    @staticmethod
    cdef Column from_libcudf(unique_ptr[column] libcudf_col)

    cpdef DataType type(self) noexcept
    cpdef Column child(self, size_type index) noexcept
    cpdef size_type num_children(self) noexcept
