# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.side_type cimport side_type
from pylibcudf.libcudf.strings.strip cimport strip as cpp_strip

from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar
import pylibcudf as plc


@acquire_spill_lock()
def strip(Column source_strings,
          object py_repl):
    """
    Returns a Column by removing leading and trailing characters.
    The set of characters need be stripped from left and right side
    can be specified by `py_repl`.
    """

    cdef DeviceScalar repl = py_repl.device_value
    return Column.from_pylibcudf(
        plc.strings.strip.strip(
            source_strings.to_pylibcudf(mode="read"),
            plc.strings.SideType.BOTH,
            repl.c_value
        )
    )


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
