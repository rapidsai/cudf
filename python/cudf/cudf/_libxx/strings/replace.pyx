# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.move cimport move
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.scalar.scalar cimport string_scalar
from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.cpp.column.column cimport column

from cudf._libxx.cpp.strings.replace cimport (
    replace_slice as cpp_replace_slice
)

from cudf._libxx.cpp.strings.substring cimport (
    slice_strings as cpp_slice_strings
)


def slice_replace(Column source_strings,
                  size_type start,
                  size_type stop,
                  Scalar repl):
    """
    Returns a Column by replacing specified section
    of each string with `repl`. Positions can be
    specified with `start` and `stop` params.
    """

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_str = <string_scalar*>(repl.c_value.get())

    with nogil:
        c_result = move(cpp_replace_slice(
            source_view,
            scalar_str[0],
            start,
            stop
        ))

    return Column.from_unique_ptr(move(c_result))


def insert(Column source_strings,
           size_type start,
           Scalar repl):
    """
    Returns a Column by inserting a specified
    string `repl` at a specific position in all strings.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_str = <string_scalar*>(repl.c_value.get())

    with nogil:
        c_result = move(cpp_replace_slice(
            source_view,
            scalar_str[0],
            start,
            start
        ))

    return Column.from_unique_ptr(move(c_result))
