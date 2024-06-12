# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.strings.attributes cimport (
    code_points as cpp_code_points,
    count_bytes as cpp_count_bytes,
    count_characters as cpp_count_characters,
)


@acquire_spill_lock()
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


@acquire_spill_lock()
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


@acquire_spill_lock()
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
