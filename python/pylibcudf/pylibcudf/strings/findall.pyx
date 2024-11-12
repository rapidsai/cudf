# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport findall as cpp_findall
from pylibcudf.strings.regex_program cimport RegexProgram

__all__ = ["findall", "find_re"]

cpdef Column findall(Column input, RegexProgram pattern):
    """
    Returns a lists column of strings for each matching occurrence using
    the regex_program pattern within each string.

    For details, see :cpp:func:`cudf::strings::findall`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    pattern : RegexProgram
        Regex pattern

    Returns
    -------
    Column
        New lists column of strings
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_findall.findall(
            input.view(),
            pattern.c_obj.get()[0]
        )

    return Column.from_libcudf(move(c_result))


cpdef Column find_re(Column input, RegexProgram pattern):
    """
    Returns character positions where the pattern first matches
    the elements in input strings.

    For details, see :cpp:func:`cudf::strings::find_re`

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    pattern : RegexProgram
        Regex pattern

    Returns
    -------
    Column
        New column of integers
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_findall.find_re(
            input.view(),
            pattern.c_obj.get()[0]
        )

    return Column.from_libcudf(move(c_result))
