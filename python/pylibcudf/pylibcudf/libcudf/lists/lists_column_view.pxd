# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/lists/lists_column_view.hpp" namespace "cudf" nogil:
    cdef cppclass lists_column_view(column_view):
        lists_column_view() except +
        lists_column_view(const lists_column_view& lists_column) except +
        lists_column_view(const column_view& lists_column) except +
        lists_column_view& operator=(const lists_column_view&) except +
        column_view parent() except +
        column_view offsets() except +
        column_view child() except +

    cdef enum:
        offsets_column_index "cudf::lists_column_view::offsets_column_index"
        child_column_index "cudf::lists_column_view::child_column_index"
