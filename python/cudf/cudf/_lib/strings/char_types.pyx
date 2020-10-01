# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.column cimport Column
from cudf._lib.scalar cimport Scalar
from cudf._lib.cpp.column.column cimport column

from cudf._lib.cpp.strings.char_types cimport (
    all_characters_of_type as cpp_all_characters_of_type,
    filter_characters_of_type as cpp_filter_characters_of_type,
    string_character_types as string_character_types,
    is_integer as cpp_is_integer,
    is_float as cpp_is_float,
)


def filter_alphanum(Column source_strings, Scalar repl, bool keep=True):
    """
    Returns a Column of strings keeping only alphanumeric character types.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef string_scalar* scalar_repl = <string_scalar*>(repl.c_value.get())

    with nogil:
        c_result = move(cpp_filter_characters_of_type(
            source_view,
            string_character_types.ALL_TYPES if keep
            else string_character_types.ALPHANUM,
            scalar_repl[0],
            string_character_types.ALPHANUM if keep
            else string_character_types.ALL_TYPES
        ))

    return Column.from_unique_ptr(move(c_result))


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
            string_character_types.DECIMAL,
            string_character_types.ALL_TYPES
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
            string_character_types.ALPHANUM,
            string_character_types.ALL_TYPES
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
            string_character_types.ALPHA,
            string_character_types.ALL_TYPES
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
            string_character_types.DIGIT,
            string_character_types.ALL_TYPES
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
            string_character_types.NUMERIC,
            string_character_types.ALL_TYPES
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
            string_character_types.UPPER,
            string_character_types.CASE_TYPES
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
            string_character_types.LOWER,
            string_character_types.CASE_TYPES
        ))

    return Column.from_unique_ptr(move(c_result))


def is_space(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contains all characters which are spaces only.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_all_characters_of_type(
            source_view,
            string_character_types.SPACE,
            string_character_types.ALL_TYPES
        ))

    return Column.from_unique_ptr(move(c_result))


def is_integer(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that have intergers.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_is_integer(
            source_view
        ))

    return Column.from_unique_ptr(move(c_result))


def is_float(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that have floats.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_is_float(
            source_view
        ))

    return Column.from_unique_ptr(move(c_result))
