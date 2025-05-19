# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport attributes as cpp_attributes

__all__ = ["code_points", "count_bytes", "count_characters"]

cpdef Column count_characters(Column source_strings):
    """
    Returns a column containing character lengths of each string
    in the given column.

    Parameters
    ----------
    source_strings : Column
        Column of strings.

    Returns
    -------
    Column
        New column with lengths for each string
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_attributes.count_characters(source_strings.view())

    return Column.from_libcudf(move(c_result))


cpdef Column count_bytes(Column source_strings):
    """
    Returns a column containing byte lengths of each string
    in the given column.

    Parameters
    ----------
    source_strings : Column
        Column of strings.

    Returns
    -------
    Column
        New column with the number of bytes for each string
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_attributes.count_bytes(source_strings.view())

    return Column.from_libcudf(move(c_result))


cpdef Column code_points(Column source_strings):
    """
    Creates a numeric column with code point values (integers)
    for each character of each string.

    Parameters
    ----------
    source_strings : Column
        Column of strings.

    Returns
    -------
    Column
        New column with code point integer values for each character
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_attributes.code_points(source_strings.view())

    return Column.from_libcudf(move(c_result))
