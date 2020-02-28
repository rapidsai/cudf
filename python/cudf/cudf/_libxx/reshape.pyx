# Copyright (c) 2019, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from cudf._libxx.move cimport move

from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view

from cudf._libxx.cpp.reshape cimport (
    interleave_columns as cpp_interleave_columns,
    tile as cpp_tile
)

def interleave_columns(Table source_table):
    cdef table_view c_view = source_table.data_view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_interleave_columns(c_view))

    return Column.from_unique_ptr(
        move(c_result)
    )


def tile(Table source_table, size_type count):
    cdef size_type c_count = count
    cdef table_view c_view = source_table.view()
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_tile(c_view, c_count))
        
    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=source_table._index_names
    )
