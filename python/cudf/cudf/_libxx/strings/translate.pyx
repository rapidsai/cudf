# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.move cimport move

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.strings.translate cimport (
    translate as cpp_translate
)
from cudf._libxx.column cimport Column
from collections import OrderedDict
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cudf._libxx.cpp.types cimport char_utf8


def translate(Column source_strings,
              object mapping_table):
    """
    Translates individual characters within each string
    mapping present in the mapping_table.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    mapping_table_dict = OrderedDict()

    for key in mapping_table:
        value = mapping_table[key]
        if type(value) is str:
            value = ord(value)
        if type(key) is str:
            key = ord(key)
        mapping_table_dict[(key, value)] = None

    cdef vector[pair[char_utf8, char_utf8]] c_mapping_table
    c_mapping_table = list(mapping_table_dict.keys())

    with nogil:
        c_result = move(cpp_translate(source_view, c_mapping_table))

    return Column.from_unique_ptr(move(c_result))
