# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.reshape cimport (
    interleave_columns as cpp_interleave_columns,
    tile as cpp_tile,
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.utils cimport columns_from_unique_ptr, table_view_from_columns


def interleave_columns(list source_columns):
    cdef table_view c_view = table_view_from_columns(source_columns)
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_interleave_columns(c_view))

    return Column.from_unique_ptr(move(c_result))


def tile(list source_columns, size_type count):
    cdef size_type c_count = count
    cdef table_view c_view = table_view_from_columns(source_columns)
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_tile(c_view, c_count))

    return columns_from_unique_ptr(move(c_result))
