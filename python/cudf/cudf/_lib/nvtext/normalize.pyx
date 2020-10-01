# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.normalize cimport (
    normalize_characters as cpp_normalize_characters,
    normalize_spaces as cpp_normalize_spaces
)
from cudf._lib.column cimport Column


def normalize_spaces(Column strings):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_normalize_spaces(c_strings))

    return Column.from_unique_ptr(move(c_result))


def normalize_characters(Column strings, bool do_lower=True):
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_normalize_characters(c_strings, do_lower))

    return Column.from_unique_ptr(move(c_result))
