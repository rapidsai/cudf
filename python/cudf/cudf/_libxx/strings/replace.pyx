# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *
from cudf._libxx.lib import *
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.scalar import Scalar

from cudf._libxx.strings.replace cimport (
    replace_slice as cpp_replace_slice
)

from cudf._libxx.strings.substring cimport (
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
