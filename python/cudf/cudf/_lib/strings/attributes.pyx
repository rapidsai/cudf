# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.strings.attributes cimport (
    count_characters as cpp_count_characters,
    code_points as cpp_code_points,
    count_bytes as cpp_count_bytes
)
from cudf._lib.column cimport Column


def count_characters(Column source_strings):
    """
    Returns an integer numeric column containing the
    length of each string in characters.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_count_characters(source_view))

    return Column.from_unique_ptr(move(c_result))


def count_bytes(Column source_strings):
    """
    Returns an integer numeric column containing the
    number of bytes of each string.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_count_bytes(source_view))

    return Column.from_unique_ptr(move(c_result))


def code_points(Column source_strings):
    """
    Creates a numeric column with code point values (integers)
    for each character of each string.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_code_points(source_view))

    return Column.from_unique_ptr(move(c_result))
