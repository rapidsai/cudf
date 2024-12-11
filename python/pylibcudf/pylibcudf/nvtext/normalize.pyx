# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.nvtext.normalize cimport (
    normalize_characters as cpp_normalize_characters,
    normalize_spaces as cpp_normalize_spaces,
)

__all__ = ["normalize_characters", "normalize_spaces"]

cpdef Column normalize_spaces(Column input):
    """
    Returns a new strings column by normalizing the whitespace in
    each string in the input column.

    For details, see :cpp:func:`normalize_spaces`

    Parameters
    ----------
    input : Column
        Input strings

    Returns
    -------
    Column
        New strings columns of normalized strings.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_normalize_spaces(input.view())

    return Column.from_libcudf(move(c_result))


cpdef Column normalize_characters(Column input, bool do_lower_case):
    """
    Normalizes strings characters for tokenizing.

    For details, see :cpp:func:`normalize_characters`

    Parameters
    ----------
    input : Column
        Input strings
    do_lower_case : bool
        If true, upper-case characters are converted to lower-case
        and accents are stripped from those characters. If false,
        accented and upper-case characters are not transformed.

    Returns
    -------
    Column
        Normalized strings column
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_normalize_characters(input.view(), do_lower_case)

    return Column.from_libcudf(move(c_result))
