# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport Scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.table cimport Table

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport string_scalar

from cudf._lib.cpp.strings.findall cimport (
    findall_re as cpp_findall_re
)
from libcpp.string cimport string


def findall(Column source_strings, pattern):
    """
    Returns a Table with all non-overlapping matches of `pattern`
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

    return Table.from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )
