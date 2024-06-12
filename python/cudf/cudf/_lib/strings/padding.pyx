# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.types cimport size_type

from enum import IntEnum

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.strings.padding cimport (
    pad as cpp_pad,
    zfill as cpp_zfill,
)
from cudf._lib.pylibcudf.libcudf.strings.side_type cimport (
    side_type,
    underlying_type_t_side_type,
)


class SideType(IntEnum):
    LEFT = <underlying_type_t_side_type> side_type.LEFT
    RIGHT = <underlying_type_t_side_type> side_type.RIGHT
    BOTH = <underlying_type_t_side_type> side_type.BOTH


@acquire_spill_lock()
def pad(Column source_strings,
        size_type width,
        fill_char,
        side=SideType.LEFT):
    """
    Returns a Column by padding strings in `source_strings`
    up to the given `width`. Direction of padding is to be specified by `side`.
    The additional characters being filled can be changed by specifying
    `fill_char`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string f_char = <string>str(fill_char).encode()

    cdef side_type pad_direction = <side_type>(
        <underlying_type_t_side_type> side
    )

    with nogil:
        c_result = move(cpp_pad(
            source_view,
            width,
            pad_direction,
            f_char
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def zfill(Column source_strings,
          size_type width):
    """
    Returns a Column by prepending strings in `source_strings`
    with '0' characters up to the given `width`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_zfill(
            source_view,
            width
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def center(Column source_strings,
           size_type width,
           fill_char):
    """
    Returns a Column by filling left and right side of strings
    in `source_strings` with additional character, `fill_char`
    up to the given `width`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string f_char = <string>str(fill_char).encode()

    with nogil:
        c_result = move(cpp_pad(
            source_view,
            width,
            side_type.BOTH,
            f_char
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def ljust(Column source_strings,
          size_type width,
          fill_char):
    """
    Returns a Column by filling right side of strings in `source_strings`
    with additional character, `fill_char` up to the given `width`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string f_char = <string>str(fill_char).encode()

    with nogil:
        c_result = move(cpp_pad(
            source_view,
            width,
            side_type.RIGHT,
            f_char
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def rjust(Column source_strings,
          size_type width,
          fill_char):
    """
    Returns a Column by filling left side of strings in `source_strings`
    with additional character, `fill_char` up to the given `width`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string f_char = <string>str(fill_char).encode()

    with nogil:
        c_result = move(cpp_pad(
            source_view,
            width,
            side_type.LEFT,
            f_char
        ))

    return Column.from_unique_ptr(move(c_result))
