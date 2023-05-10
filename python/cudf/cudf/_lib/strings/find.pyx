# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.strings.find cimport (
    contains as cpp_contains,
    ends_with as cpp_ends_with,
    find as cpp_find,
    rfind as cpp_rfind,
    starts_with as cpp_starts_with,
)
from cudf._lib.cpp.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def contains(Column source_strings, object py_target):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain the pattern given in `py_target`.
    """
    cdef DeviceScalar target = py_target.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_str = <const string_scalar*>(
        target.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_contains(
            source_view,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def contains_multiple(Column source_strings, Column target_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain the corresponding string in `target_strings`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef column_view target_view = target_strings.view()

    with nogil:
        c_result = move(cpp_contains(
            source_view,
            target_view
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def endswith(Column source_strings, object py_target):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain strings that end with the pattern given in `py_target`.
    """

    cdef DeviceScalar target = py_target.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_str = <const string_scalar*>(
        target.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_ends_with(
            source_view,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
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


@acquire_spill_lock()
def startswith(Column source_strings, object py_target):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain strings that start with the pattern given in `py_target`.
    """

    cdef DeviceScalar target = py_target.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_str = <const string_scalar*>(
        target.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_starts_with(
            source_view,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
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


@acquire_spill_lock()
def find(Column source_strings,
         object py_target,
         size_type start,
         size_type end):
    """
    Returns a Column containing lowest indexes in each string of
    `source_strings` that fully contain `py_target` string.
    Scan portion of strings in `source_strings` can be
    controlled by setting `start` and `end` values.
    """

    cdef DeviceScalar target = py_target.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_str = <const string_scalar*>(
        target.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_find(
            source_view,
            scalar_str[0],
            start,
            end
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def rfind(Column source_strings,
          object py_target,
          size_type start,
          size_type end):
    """
    Returns a Column containing highest indexes in each string of
    `source_strings` that fully contain `py_target` string.
    Scan portion of strings in `source_strings` can be
    controlled by setting `start` and `end` values.
    """

    cdef DeviceScalar target = py_target.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_str = <const string_scalar*>(
        target.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_rfind(
            source_view,
            scalar_str[0],
            start,
            end
        ))

    return Column.from_unique_ptr(move(c_result))
