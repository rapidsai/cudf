# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.move cimport move
from cudf._libxx.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.types cimport size_type

from cudf._libxx.table cimport Table

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.scalar.scalar cimport string_scalar

from cudf._libxx.cpp.strings.replace_re cimport (
    replace_re as cpp_replace_re
)
from libcpp.string cimport string


def replace_re(Column source_strings, pattern, Scalar repl, size_type n):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only decimal characters -- those that can be used
    to extract base10 numbers.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string pattern_string = <string>str(pattern).encode()
    cdef string_scalar* scalar_repl = \
        <string_scalar*>(repl.c_value.get())

    with nogil:
        c_result = move(cpp_replace_re(
            source_view,
            pattern_string,
            scalar_repl[0],
            n
        ))

    return Column.from_unique_ptr(move(c_result))
