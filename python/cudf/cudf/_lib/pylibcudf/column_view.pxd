# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column_view cimport column_view


cdef class ColumnView:
    cdef unique_ptr[column_view] c_obj
    # While view types are designed to be cheap to copy and it would be
    # acceptable to return by value rather than reference here, similar helpers
    # will be necessary for owning types to simplify their internals (e.g. a
    # `pylibcudf.Column.get` method) and in those cases we will need to return
    # a pointer. Returning a pointer here is therefore preferable for symmetry
    # for symmetry.
    cdef column_view * get(self) nogil
