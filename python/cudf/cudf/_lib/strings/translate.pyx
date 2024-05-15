# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.translate cimport (
    filter_characters as cpp_filter_characters,
    filter_type,
    translate as cpp_translate,
)
from cudf._lib.pylibcudf.libcudf.types cimport char_utf8
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def translate(Column source_strings,
              object mapping_table):
    """
    Translates individual characters within each string
    if present in the mapping_table.
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


@acquire_spill_lock()
def filter_characters(Column source_strings,
                      object mapping_table,
                      bool keep,
                      object py_repl):
    """
    Removes or keeps individual characters within each string
    using the provided mapping_table.
    """

    cdef DeviceScalar repl = py_repl.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef const string_scalar* scalar_repl = <const string_scalar*>(
        repl.get_raw_ptr()
    )
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

    cdef filter_type c_keep
    if keep is True:
        c_keep = filter_type.KEEP
    else:
        c_keep = filter_type.REMOVE

    with nogil:
        c_result = move(cpp_filter_characters(
            source_view,
            c_mapping_table,
            c_keep,
            scalar_repl[0]
        ))

    return Column.from_unique_ptr(move(c_result))
