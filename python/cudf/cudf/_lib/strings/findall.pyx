# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.strings.contains cimport regex_flags
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.strings.findall cimport (
    findall as cpp_findall,
    findall_record as cpp_findall_record,
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport data_from_unique_ptr


def findall(Column source_strings, object pattern, uint32_t flags):
    """
    Returns data with all non-overlapping matches of `pattern`
    in each string of `source_strings`.
    """
    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()

    cdef string pattern_string = <string>str(pattern).encode()
    cdef regex_flags c_flags = <regex_flags>flags

    with nogil:
        c_result = move(cpp_findall(
            source_view,
            pattern_string,
            c_flags
        ))

    return data_from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )


def findall_record(Column source_strings, object pattern, uint32_t flags):
    """
    Returns data with all non-overlapping matches of `pattern`
    in each string of `source_strings` as a lists column.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string pattern_string = <string>str(pattern).encode()
    cdef regex_flags c_flags = <regex_flags>flags

    with nogil:
        c_result = move(cpp_findall_record(
            source_view,
            pattern_string,
            c_flags
        ))

    return Column.from_unique_ptr(move(c_result))
