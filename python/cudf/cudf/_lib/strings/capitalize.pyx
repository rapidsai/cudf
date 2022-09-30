# Copyright (c) 2018-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.strings.capitalize cimport (
    capitalize as cpp_capitalize,
    is_title as cpp_is_title,
    title as cpp_title,
)


def capitalize(Column source_strings):
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_capitalize(source_view))

    return Column.from_unique_ptr(move(c_result))


def title(Column source_strings):
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_title(source_view))

    return Column.from_unique_ptr(move(c_result))


def is_title(Column source_strings):
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_is_title(source_view))

    return Column.from_unique_ptr(move(c_result))
