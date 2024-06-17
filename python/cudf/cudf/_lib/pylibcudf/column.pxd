# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from cudf._lib.pylibcudf.libcudf.types cimport bitmask_type, size_type

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
        # _children: List[Column]
        list _children
        size_type _num_children

    cdef column_view view(self) nogil
    cdef mutable_column_view mutable_view(self) nogil

    @staticmethod
    cdef Column from_libcudf(unique_ptr[column] libcudf_col)

    @staticmethod
    cdef Column from_column_view(const column_view& libcudf_col, Column owner)

    cpdef DataType type(self)
    cpdef Column child(self, size_type index)
    cpdef size_type num_children(self)
    cpdef size_type size(self)
    cpdef size_type null_count(self)
    cpdef size_type offset(self)
    cpdef gpumemoryview data(self)
    cpdef gpumemoryview null_mask(self)
    cpdef list children(self)
    cpdef Column copy(self)

    cpdef ListColumnView list_view(self)


cdef class ListColumnView:
    """Accessor for methods of a Column that are specific to lists."""
    cdef Column _column
    cpdef child(self)
    cpdef offsets(self)
