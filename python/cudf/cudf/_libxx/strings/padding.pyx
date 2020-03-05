# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.move cimport move
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.scalar.scalar cimport string_scalar
from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar
from libcpp.string cimport string

from cudf._libxx.strings.padding cimport (
    pad as cpp_pad,
    zfill as cpp_zfill,
    pad_side as pad_side
)


def pad(Column source_strings,
        size_type width,
        side,
        fill_char):
    """
    Returns a Column by padding strings in `source_strings`
    upto the given `width`. Direction of padding is to be specified by `side`.
    The additional characters being filled can be changed by specifying
    `fill_char`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef pad_side pad_direction
    cdef string f_char = <string>str(fill_char).encode()

    if side == "left":
        pad_direction = pad_side.LEFT
    elif side == "right":
        pad_direction = pad_side.RIGHT
    elif side == "both":
        pad_direction = pad_side.BOTH

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
    with ‘0’ characters upto the given `width`.
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
    upto the given `width`.
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
    with additional character, `fill_char` upto the given `width`.
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


def rjust(Column source_strings,
          size_type width,
          fill_char):
    """
    Returns a Column by filling left side of strings in `source_strings`
    with additional character, `fill_char` upto the given `width`.
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
