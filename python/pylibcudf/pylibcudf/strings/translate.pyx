# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings cimport translate as cpp_translate
from pylibcudf.libcudf.types cimport char_utf8
from pylibcudf.scalar cimport Scalar

from cython.operator import dereference
from pylibcudf.libcudf.strings.translate import \
    filter_type as FilterType  # no-cython-lint

__all__ = ["FilterType", "filter_characters", "translate"]

cdef vector[pair[char_utf8, char_utf8]] _table_to_c_table(dict table):
    """
    Convert str.maketrans table to cudf compatible table.
    """
    cdef int table_size = len(table)
    cdef vector[pair[char_utf8, char_utf8]] c_table

    c_table.reserve(table_size)
    for key, value in table.items():
        if isinstance(value, int):
            value = chr(value)
        if isinstance(value, str):
            value = int.from_bytes(value.encode(), byteorder='big')
        if isinstance(key, int):
            key = chr(key)
        if isinstance(key, str):
            key = int.from_bytes(key.encode(), byteorder='big')
        c_table.push_back((key, value))

    return c_table


cpdef Column translate(Column input, dict chars_table):
    """
    Translates individual characters within each string.

    For details, see :cpp:func:`cudf::strings::translate`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation

    chars_table : dict
        Table of UTF-8 character mappings

    Returns
    -------
    Column
        New column with padded strings.
    """
    cdef unique_ptr[column] c_result
    cdef vector[pair[char_utf8, char_utf8]] c_chars_table = _table_to_c_table(
        chars_table
    )

    with nogil:
        c_result = cpp_translate.translate(
            input.view(),
            c_chars_table
        )
    return Column.from_libcudf(move(c_result))


cpdef Column filter_characters(
    Column input,
    dict characters_to_filter,
    filter_type keep_characters,
    Scalar replacement
):
    """
    Removes ranges of characters from each string in a strings column.

    For details, see :cpp:func:`cudf::strings::filter_characters`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation

    characters_to_filter : dict
        Table of character ranges to filter on

    keep_characters : FilterType
        If true, the `characters_to_filter` are retained
        and all other characters are removed.

    replacement : Scalar
        Replacement string for each character removed.

    Returns
    -------
    Column
        New column with filtered strings.
    """
    cdef unique_ptr[column] c_result
    cdef vector[pair[char_utf8, char_utf8]] c_characters_to_filter = _table_to_c_table(
        characters_to_filter
    )
    cdef const string_scalar* c_replacement = <const string_scalar*>(
        replacement.c_obj.get()
    )

    with nogil:
        c_result = cpp_translate.filter_characters(
            input.view(),
            c_characters_to_filter,
            keep_characters,
            dereference(c_replacement),
        )
    return Column.from_libcudf(move(c_result))
