# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings cimport char_types as cpp_char_types
from pylibcudf.scalar cimport Scalar

from cython.operator import dereference
from pylibcudf.libcudf.strings.char_types import \
    string_character_types as StringCharacterTypes  # no-cython-lint

__all__ = [
    "StringCharacterTypes",
    "all_characters_of_type",
    "filter_characters_of_type",
]

cpdef Column all_characters_of_type(
    Column source_strings,
    string_character_types types,
    string_character_types verify_types
):
    """
    Identifies strings where all characters match the specified type.

    Parameters
    ----------
    source_strings : Column
        Strings instance for this operation
    types : StringCharacterTypes
        The character types to check in each string
    verify_types : StringCharacterTypes
        Only verify against these character types.

    Returns
    -------
    Column
        New column of boolean results for each string
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_char_types.all_characters_of_type(
            source_strings.view(),
            types,
            verify_types,
        )

    return Column.from_libcudf(move(c_result))

cpdef Column filter_characters_of_type(
    Column source_strings,
    string_character_types types_to_remove,
    Scalar replacement,
    string_character_types types_to_keep
):
    """
    Filter specific character types from a column of strings.

    Parameters
    ----------
    source_strings : Column
        Strings instance for this operation
    types_to_remove : StringCharacterTypes
        The character types to check in each string.
    replacement : Scalar
        The replacement character to use when removing characters
    types_to_keep : StringCharacterTypes
        Default `ALL_TYPES` means all characters of `types_to_remove`
        will be filtered.

    Returns
    -------
    Column
        New column with the specified characters filtered out and
        replaced with the specified replacement string.
    """
    cdef const string_scalar* c_replacement = <const string_scalar*>(
        replacement.c_obj.get()
    )
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_char_types.filter_characters_of_type(
            source_strings.view(),
            types_to_remove,
            dereference(c_replacement),
            types_to_keep,
        )

    return Column.from_libcudf(move(c_result))
