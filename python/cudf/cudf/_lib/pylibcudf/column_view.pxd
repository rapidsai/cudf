# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type


cdef class ColumnView:
    cdef unique_ptr[column_view] c_obj
    # While view types are designed to be cheap to copy and it would be
    # acceptable to return by value rather than reference here, similar helpers
    # will be necessary for owning types to simplify their internals (e.g. a
    # `pylibcudf.Column.get` method) and in those cases we will need to return
    # a pointer. Returning a pointer here is therefore preferable for symmetry.
    cdef column_view * get(self) nogil

    @staticmethod
    cdef from_column_view(column_view cv)
    cpdef size_type size(self)
    cpdef size_type null_count(self)
