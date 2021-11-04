# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.strings.findall cimport findall_re as cpp_findall_re
from cudf._lib.cpp.table.table cimport table
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport data_from_unique_ptr


def findall(Column source_strings, pattern):
    """
    Returns data with all non-overlapping matches of `pattern`
    in each string of `source_strings`.
    """
    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()

    cdef string pattern_string = <string>str(pattern).encode()

    with nogil:
        c_result = move(cpp_findall_re(
            source_view,
            pattern_string
        ))

    return data_from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )
