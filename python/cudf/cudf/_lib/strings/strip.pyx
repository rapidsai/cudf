# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport Scalar
from libcpp.string cimport string
from cudf._lib.cpp.column.column cimport column

from cudf._lib.cpp.strings.strip cimport (
    strip as cpp_strip,
    strip_type as strip_type
)


def strip(Column source_strings,
          Scalar repl):
    """
    Returns a Column by removing leading and trailing characters.
    The set of characters need be stripped from left and right side
    can be specified by `repl`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_str = <string_scalar*>(repl.c_value.get())

    with nogil:
        c_result = move(cpp_strip(
            source_view,
            strip_type.BOTH,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def lstrip(Column source_strings,
           Scalar repl):
    """
    Returns a Column by removing leading and trailing characters.
    The set of characters need be stripped from left side can
    be specified by `repl`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_str = <string_scalar*>(repl.c_value.get())

    with nogil:
        c_result = move(cpp_strip(
            source_view,
            strip_type.LEFT,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def rstrip(Column source_strings,
           Scalar repl):
    """
    Returns a Column by removing leading and trailing characters.
    The set of characters need be stripped from right side can
    be specified by `repl`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_str = <string_scalar*>(repl.c_value.get())

    with nogil:
        c_result = move(cpp_strip(
            source_view,
            strip_type.RIGHT,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))
