# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.side_type cimport side_type
from cudf._lib.pylibcudf.libcudf.strings.strip cimport strip as cpp_strip
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def strip(Column source_strings,
          object py_repl):
    """
    Returns a Column by removing leading and trailing characters.
    The set of characters need be stripped from left and right side
    can be specified by `py_repl`.
    """

    cdef DeviceScalar repl = py_repl.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_str = <const string_scalar*>(
        repl.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_strip(
            source_view,
            side_type.BOTH,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def lstrip(Column source_strings,
           object py_repl):
    """
    Returns a Column by removing leading and trailing characters.
    The set of characters need be stripped from left side can
    be specified by `py_repl`.
    """

    cdef DeviceScalar repl = py_repl.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_str = <const string_scalar*>(
        repl.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_strip(
            source_view,
            side_type.LEFT,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def rstrip(Column source_strings,
           object py_repl):
    """
    Returns a Column by removing leading and trailing characters.
    The set of characters need be stripped from right side can
    be specified by `py_repl`.
    """

    cdef DeviceScalar repl = py_repl.device_value

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef const string_scalar* scalar_str = <const string_scalar*>(
        repl.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_strip(
            source_view,
            side_type.RIGHT,
            scalar_str[0]
        ))

    return Column.from_unique_ptr(move(c_result))
