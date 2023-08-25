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
        DataType _data_type
        size_type _size
        gpumemoryview _data
        gpumemoryview _mask
        size_type _null_count
        size_type _offset
        # children: List[Column]
        list _children
        size_type _num_children

    cdef column_view view(self) nogil

    @staticmethod
    cdef Column from_libcudf(unique_ptr[column] libcudf_col)

    cpdef DataType type(self)
    cpdef Column child(self, size_type index)
    cpdef size_type num_children(self)
    cpdef size_type size(self)
    cpdef size_type null_count(self)
    cpdef size_type offset(self)
    cpdef gpumemoryview data(self)
    cpdef gpumemoryview null_mask(self)
    cpdef list children(self)

    cpdef list_view(self)


cdef class ListColumnView:
    """Accessor for methods of a Column that are specific to lists."""
    cdef Column _column
    cpdef child(self)
    cpdef offsets(self)
