# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.move cimport move
from cudf._libxx.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.scalar.scalar cimport string_scalar
from cudf._libxx.cpp.types cimport size_type

from cudf._libxx.cpp.strings.find cimport (
    contains as cpp_contains,
    ends_with as cpp_ends_with,
    starts_with as cpp_starts_with,
    find as cpp_find,
    rfind as cpp_rfind
)


def contains(Column source_strings, Scalar target):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain the pattern given in `target`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_str = <string_scalar*>(target.c_value.get())

    with nogil:
        c_result = move(cpp_contains(
            source_view,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def endswith(Column source_strings, Scalar target):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain strings that end with the pattern given in `target`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_str = <string_scalar*>(target.c_value.get())

    with nogil:
        c_result = move(cpp_ends_with(
            source_view,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def startswith(Column source_strings, Scalar target):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain strings that start with the pattern given in `target`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_str = <string_scalar*>(target.c_value.get())

    with nogil:
        c_result = move(cpp_starts_with(
            source_view,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def find(Column source_strings,
         Scalar target,
         size_type start,
         size_type end):
    """

    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_str = <string_scalar*>(target.c_value.get())

    with nogil:
        c_result = move(cpp_find(
            source_view,
            scalar_str[0],
            start,
            end
        ))

    return Column.from_unique_ptr(move(c_result))


def rfind(Column source_strings,
          Scalar target,
          size_type start,
          size_type end):
    """

    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* scalar_str = <string_scalar*>(target.c_value.get())

    with nogil:
        c_result = move(cpp_rfind(
            source_view,
            scalar_str[0],
            start,
            end
        ))

    return Column.from_unique_ptr(move(c_result))
