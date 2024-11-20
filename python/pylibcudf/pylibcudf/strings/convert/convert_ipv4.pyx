# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport convert_ipv4 as cpp_convert_ipv4

__all__ = ["integers_to_ipv4", "ipv4_to_integers", "is_ipv4"]

cpdef Column ipv4_to_integers(Column input):
    """
    Converts IPv4 addresses into integers.

    For details, see cpp:func:`cudf::strings::ipv4_to_integers`

    Parameters
    ----------
    input : Column
        Strings instance for this operation

    Returns
    -------
    Column
        New uint32 column converted from strings.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_convert_ipv4.ipv4_to_integers(input.view())

    return Column.from_libcudf(move(c_result))


cpdef Column integers_to_ipv4(Column integers):
    """
    Converts integers into IPv4 addresses as strings.

    For details, see cpp:func:`cudf::strings::integers_to_ipv4`

    Parameters
    ----------
    integers : Column
        Integer (uint32) column to convert.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_convert_ipv4.integers_to_ipv4(integers.view())

    return Column.from_libcudf(move(c_result))


cpdef Column is_ipv4(Column input):
    """
    Returns a boolean column identifying strings in which all
    characters are valid for conversion to integers from IPv4 format.

    For details, see cpp:func:`cudf::strings::is_ipv4`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    Returns
    -------
    Column
        New column of boolean results for each string.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_convert_ipv4.is_ipv4(input.view())

    return Column.from_libcudf(move(c_result))
