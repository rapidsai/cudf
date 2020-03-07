# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.move cimport move
from cudf._libxx.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.scalar.scalar cimport string_scalar

from cudf._libxx.cpp.strings.contains cimport (
    contains_re as cpp_contains_re
)
from libcpp.string cimport string


def contains_re(Column source_strings, reg_ex):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only decimal characters -- those that can be used
    to extract base10 numbers.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string reg_ex_string = <string>str(reg_ex).encode()

    with nogil:
        c_result = move(cpp_contains_re(
            source_view,
            reg_ex_string
        ))

    return Column.from_unique_ptr(move(c_result))
