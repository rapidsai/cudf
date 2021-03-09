# Copyright (c) 2019-2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cudf._lib.column cimport Column
from cudf._lib.table cimport Table

from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view

from cudf._lib.cpp.reshape cimport (
    interleave_columns as cpp_interleave_columns,
    tile as cpp_tile,
    explode as cpp_explode
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


def explode(Table input_table, explode_column_name, ignore_index, nlevels):
    cdef table_view c_table_view = \
        input_table.data_view() if ignore_index else input_table.view()
    cdef size_type c_column_idx = \
        input_table._column_names.index(explode_column_name)
    if not ignore_index:
        c_column_idx += nlevels
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_explode(c_table_view, c_column_idx))

    exploded_index_names = None if ignore_index else input_table._index_names

    return Table.from_unique_ptr(
        move(c_result),
        column_names=input_table._column_names,
        index_names=exploded_index_names
    )
