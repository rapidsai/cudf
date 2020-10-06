# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport Scalar
from enum import IntEnum
from libcpp.string cimport string
from cudf._lib.cpp.column.column cimport column

from cudf._lib.cpp.strings.padding cimport (
    pad as cpp_pad,
    zfill as cpp_zfill,
    pad_side as pad_side
)
from cudf._lib.strings.padding cimport underlying_type_t_pad_side


class PadSide(IntEnum):
    LEFT = <underlying_type_t_pad_side> pad_side.LEFT
    RIGHT = <underlying_type_t_pad_side> pad_side.RIGHT
    BOTH = <underlying_type_t_pad_side> pad_side.BOTH


def pad(Column source_strings,
        size_type width,
        fill_char,
        side=PadSide.LEFT):
    """
    Returns a Column by padding strings in `source_strings`
    up to the given `width`. Direction of padding is to be specified by `side`.
    The additional characters being filled can be changed by specifying
    `fill_char`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string f_char = <string>str(fill_char).encode()

    cdef pad_side pad_direction = <pad_side>(
        <underlying_type_t_pad_side> side
    )

    with nogil:
        c_result = move(cpp_pad(
            source_view,
            width,
            pad_direction,
            f_char
        ))

    return Column.from_unique_ptr(move(c_result))


def zfill(Column source_strings,
          size_type width):
    """
    Returns a Column by prepending strings in `source_strings`
    with ‘0’ characters up to the given `width`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_zfill(
            source_view,
            width
        ))

    return Column.from_unique_ptr(move(c_result))


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

    cdef pad_side pad_direction
    cdef string f_char = <string>str(fill_char).encode()

    with nogil:
        c_result = move(cpp_pad(
            source_view,
            width,
            pad_side.BOTH,
            f_char
        ))

    return Column.from_unique_ptr(move(c_result))


def ljust(Column source_strings,
          size_type width,
          fill_char):
    """
    Returns a Column by filling right side of strings in `source_strings`
    with additional character, `fill_char` up to the given `width`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef pad_side pad_direction
    cdef string f_char = <string>str(fill_char).encode()

    with nogil:
        c_result = move(cpp_pad(
            source_view,
            width,
            pad_side.RIGHT,
            f_char
        ))

    return Column.from_unique_ptr(move(c_result))


def rjust(Column source_strings,
          size_type width,
          fill_char):
    """
    Returns a Column by filling left side of strings in `source_strings`
    with additional character, `fill_char` up to the given `width`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef pad_side pad_direction
    cdef string f_char = <string>str(fill_char).encode()

    with nogil:
        c_result = move(cpp_pad(
            source_view,
            width,
            pad_side.LEFT,
            f_char
        ))

    return Column.from_unique_ptr(move(c_result))
