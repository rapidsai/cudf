# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.move cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.strings.translate cimport (
    translate as cpp_translate
)
from cudf._lib.column cimport Column
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cudf._lib.cpp.types cimport char_utf8


def translate(Column source_strings,
              object mapping_table):
    """
    Translates individual characters within each string
    mapping present in the mapping_table.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef int table_size
    table_size = len(mapping_table)

    cdef vector[pair[char_utf8, char_utf8]] c_mapping_table
    c_mapping_table.reserve(table_size)

    for key in mapping_table:
        value = mapping_table[key]
        if type(value) is int:
            value = chr(value)
        if type(value) is str:
            value = int.from_bytes(value.encode(), byteorder='big')
        if type(key) is int:
            key = chr(key)
        if type(key) is str:
            key = int.from_bytes(key.encode(), byteorder='big')
        c_mapping_table.push_back((key, value))

    with nogil:
        c_result = move(cpp_translate(source_view, c_mapping_table))

    return Column.from_unique_ptr(move(c_result))
