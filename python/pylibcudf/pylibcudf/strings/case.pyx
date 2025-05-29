# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport case as cpp_case

__all__ = ["swapcase", "to_lower", "to_upper"]

cpdef Column to_lower(Column input):
    """Returns a column of lowercased strings.

    For details, see :cpp:func:`cudf::strings::to_lower`.

    Parameters
    ----------
    input : Column
        String column

    Returns
    -------
    pylibcudf.Column
        Column of strings lowercased from the input column
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_case.to_lower(input.view())

    return Column.from_libcudf(move(c_result))

cpdef Column to_upper(Column input):
    """Returns a column of uppercased strings.

    For details, see :cpp:func:`cudf::strings::to_upper`.

    Parameters
    ----------
    input : Column
        String column

    Returns
    -------
    pylibcudf.Column
        Column of strings uppercased from the input column
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_case.to_upper(input.view())

    return Column.from_libcudf(move(c_result))

cpdef Column swapcase(Column input):
    """Returns a column of strings where the lowercase characters
    are converted to uppercase and the uppercase characters
    are converted to lowercase.

    For details, see :cpp:func:`cudf::strings::swapcase`.

    Parameters
    ----------
    input : Column
        String column

    Returns
    -------
    pylibcudf.Column
        Column of strings
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_case.swapcase(input.view())

    return Column.from_libcudf(move(c_result))
