# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.cpp.column.column cimport column

from libc.stdint cimport int32_t

from cudf._lib.cpp.strings.replace cimport (
    replace_slice as cpp_replace_slice,
    replace as cpp_replace
)

from cudf._lib.cpp.strings.substring cimport (
    slice_strings as cpp_slice_strings
)


def slice_replace(Column source_strings,
                  size_type start,
                  size_type stop,
                  object py_repl):
    """
    Returns a Column by replacing specified section
    of each string with `py_repl`. Positions can be
    specified with `start` and `stop` params.
    """

    cdef DeviceScalar repl = py_repl.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_str = <const string_scalar*>(
        repl.get_raw_ptr()
    )

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
           object py_repl):
    """
    Returns a Column by inserting a specified
    string `py_repl` at a specific position in all strings.
    """

    cdef DeviceScalar repl = py_repl.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_str = <const string_scalar*>(
        repl.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_replace_slice(
            source_view,
            scalar_str[0],
            start,
            start
        ))

    return Column.from_unique_ptr(move(c_result))


def replace(Column source_strings,
            object py_target,
            object py_repl,
            int32_t maxrepl):
    """
    Returns a Column after replacing occurrences of
    patterns `py_target` with `py_repl` in `source_strings`.
    `maxrepl` indicates number of replacements to make from start.
    """
    cdef DeviceScalar target = py_target.device_value
    cdef DeviceScalar repl = py_repl.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_target = <const string_scalar*>(
        target.get_raw_ptr()
    )
    cdef const string_scalar* scalar_repl = <const string_scalar*>(
        repl.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_replace(
            source_view,
            scalar_target[0],
            scalar_repl[0],
            maxrepl
        ))

    return Column.from_unique_ptr(move(c_result))


def replace_multi(Column source_strings,
                  Column target_strings,
                  Column repl_strings):
    """
    Returns a Column after replacing occurrences of
    patterns `target_strings` with `repl_strings` in `source_strings`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef column_view target_view = target_strings.view()
    cdef column_view repl_view = repl_strings.view()

    with nogil:
        c_result = move(cpp_replace(
            source_view,
            target_view,
            repl_view
        ))

    return Column.from_unique_ptr(move(c_result))
