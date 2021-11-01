# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.strings.contains cimport (
    contains_re as cpp_contains_re,
    count_re as cpp_count_re,
    matches_re as cpp_matches_re,
    regex_flags as regex_flags,
)
from cudf._lib.scalar cimport DeviceScalar


def contains_re(Column source_strings, object reg_ex, uint32_t flags):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain regular expression `reg_ex`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string reg_ex_string = <string>str(reg_ex).encode()
    cdef regex_flags c_flags = <regex_flags>flags

    with nogil:
        c_result = move(cpp_contains_re(
            source_view,
            reg_ex_string,
            c_flags
        ))

    return Column.from_unique_ptr(move(c_result))


def count_re(Column source_strings, object reg_ex, uint32_t flags):
    """
    Returns a Column with count of occurrences of `reg_ex` in
    each string of `source_strings`
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string reg_ex_string = <string>str(reg_ex).encode()
    cdef regex_flags c_flags = <regex_flags>flags

    with nogil:
        c_result = move(cpp_count_re(
            source_view,
            reg_ex_string,
            c_flags
        ))

    return Column.from_unique_ptr(move(c_result))


def match_re(Column source_strings, object reg_ex, uint32_t flags):
    """
    Returns a Column with each value True if the string matches `reg_ex`
    regular expression with each record of `source_strings`
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string reg_ex_string = <string>str(reg_ex).encode()
    cdef regex_flags c_flags = <regex_flags>flags

    with nogil:
        c_result = move(cpp_matches_re(
            source_view,
            reg_ex_string,
            c_flags
        ))

    return Column.from_unique_ptr(move(c_result))
