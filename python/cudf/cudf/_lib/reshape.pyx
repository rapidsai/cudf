# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move, pair

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.reshape cimport (
    interleave_columns as cpp_interleave_columns,
    tile as cpp_tile,
    one_hot_encoding as cpp_one_hot_encoding
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.table cimport Table
from cudf._lib.utils cimport data_from_unique_ptr


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

    return data_from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=source_table._index_names
    )

def one_hot_encoding(Column input_column):
    cdef column_view c_view = input_column.view()
    cdef pair[unique_ptr[column], unique_ptr[table]] c_result

    with nogil:
        c_result = move(cpp_one_hot_encoding(c_view))
    
    column_labels = Column.from_unique_ptr(move(c_result.first))
    encodings = data_from_unique_ptr(move(c_result.second), 
        column_names=column_labels.to_pandas())

    return encodings
