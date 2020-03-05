# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.move cimport move
from cudf._libxx.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from cudf._libxx.column cimport Column

from cudf._libxx.strings.char_types cimport (
    all_characters_of_type as cpp_all_characters_of_type,
    string_character_types as string_character_types
)


def is_decimal(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only decimal characters -- those that can be used
    to extract base10 numbers.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_all_characters_of_type(
            source_view,
            string_character_types.DECIMAL
        ))

    return Column.from_unique_ptr(move(c_result))


def is_alnum(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only alpha-numeric characters.

    Equivalent to: is_alpha() or is_digit() or is_numeric() or is_decimal()
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_all_characters_of_type(
            source_view,
            string_character_types.ALPHANUM
        ))

    return Column.from_unique_ptr(move(c_result))


def is_alpha(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only alphabetic characters.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_all_characters_of_type(
            source_view,
            string_character_types.ALPHA
        ))

    return Column.from_unique_ptr(move(c_result))


def is_digit(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only decimal and digit characters.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_all_characters_of_type(
            source_view,
            string_character_types.DIGIT
        ))

    return Column.from_unique_ptr(move(c_result))


def is_numeric(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only numeric characters. These include digit and
    numeric characters.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_all_characters_of_type(
            source_view,
            string_character_types.NUMERIC
        ))

    return Column.from_unique_ptr(move(c_result))


def is_upper(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only upper-case characters.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_all_characters_of_type(
            source_view,
            string_character_types.UPPER
        ))

    return Column.from_unique_ptr(move(c_result))


def is_lower(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only lower-case characters.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_all_characters_of_type(
            source_view,
            string_character_types.LOWER
        ))

    return Column.from_unique_ptr(move(c_result))
