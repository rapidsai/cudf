# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.move cimport move

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.strings.capitalize cimport (
    capitalize as cpp_capitalize,
    title as cpp_title,
)
from cudf._libxx.column cimport Column


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
