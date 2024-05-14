# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libc.stdint cimport uint32_t

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.contains cimport (
    contains_re as cpp_contains_re,
    count_re as cpp_count_re,
    like as cpp_like,
    matches_re as cpp_matches_re,
)
from cudf._lib.pylibcudf.libcudf.strings.regex_flags cimport regex_flags
from cudf._lib.pylibcudf.libcudf.strings.regex_program cimport regex_program
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def contains_re(Column source_strings, object reg_ex, uint32_t flags):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain regular expression `reg_ex`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string reg_ex_string = <string>str(reg_ex).encode()
    cdef regex_flags c_flags = <regex_flags>flags
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(reg_ex_string, c_flags))
        c_result = move(cpp_contains_re(
            source_view,
            dereference(c_prog)
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def count_re(Column source_strings, object reg_ex, uint32_t flags):
    """
    Returns a Column with count of occurrences of `reg_ex` in
    each string of `source_strings`
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string reg_ex_string = <string>str(reg_ex).encode()
    cdef regex_flags c_flags = <regex_flags>flags
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(reg_ex_string, c_flags))
        c_result = move(cpp_count_re(
            source_view,
            dereference(c_prog)
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def match_re(Column source_strings, object reg_ex, uint32_t flags):
    """
    Returns a Column with each value True if the string matches `reg_ex`
    regular expression with each record of `source_strings`
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string reg_ex_string = <string>str(reg_ex).encode()
    cdef regex_flags c_flags = <regex_flags>flags
    cdef unique_ptr[regex_program] c_prog

    with nogil:
        c_prog = move(regex_program.create(reg_ex_string, c_flags))
        c_result = move(cpp_matches_re(
            source_view,
            dereference(c_prog)
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def like(Column source_strings, object py_pattern, object py_escape):
    """
    Returns a Column with each value True if the string matches the
    `py_pattern` like expression with each record of `source_strings`
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef DeviceScalar pattern = py_pattern.device_value
    cdef DeviceScalar escape = py_escape.device_value

    cdef const string_scalar* scalar_ptn = <const string_scalar*>(
        pattern.get_raw_ptr()
    )
    cdef const string_scalar* scalar_esc = <const string_scalar*>(
        escape.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_like(
            source_view,
            scalar_ptn[0],
            scalar_esc[0]
        ))

    return Column.from_unique_ptr(move(c_result))
