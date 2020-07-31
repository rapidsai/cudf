# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column_view cimport (
    column_view, mutable_column_view
)


cdef extern from "cudf/lists/lists_column_view.hpp" namespace "cudf" nogil:
    cdef cppclass lists_column_view(column_view):
        lists_column_view() except +
        lists_column_view(const column_view& lists_column) except +
        column_view parent() except +
        column_view offsets() except +
        column_view child() except +
