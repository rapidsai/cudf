# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport Scalar
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type

from cudf._lib.cpp.strings.find cimport (
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


def endswith_multiple(Column source_strings, Column target_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain strings that end with corresponding location
    in `target_strings`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef column_view target_view = target_strings.view()

    with nogil:
        c_result = move(cpp_ends_with(
            source_view,
            target_view
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


def startswith_multiple(Column source_strings, Column target_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain strings that begin with corresponding location
    in `target_strings`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef column_view target_view = target_strings.view()

    with nogil:
        c_result = move(cpp_starts_with(
            source_view,
            target_view
        ))

    return Column.from_unique_ptr(move(c_result))


def find(Column source_strings,
         Scalar target,
         size_type start,
         size_type end):
    """
    Returns a Column containing lowest indexes in each string of
    `source_strings` that fully contain `target` string.
    Scan portion of strings in `source_strings` can be
    controlled by setting `start` and `end` values.
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
    Returns a Column containing highest indexes in each string of
    `source_strings` that fully contain `target` string.
    Scan portion of strings in `source_strings` can be
    controlled by setting `start` and `end` values.
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
