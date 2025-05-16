# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport padding as cpp_padding
from pylibcudf.libcudf.strings.side_type cimport side_type

__all__ = ["pad", "zfill"]

cpdef Column pad(Column input, size_type width, side_type side, str fill_char):
    """
    Add padding to each string using a provided character.

    For details, see :cpp:func:`cudf::strings::pad`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    width : int
        The minimum number of characters for each string.
    side : SideType
        Where to place the padding characters.
    fill_char : str
        Single UTF-8 character to use for padding

    Returns
    -------
    Column
        New column with padded strings.
    """
    cdef unique_ptr[column] c_result
    cdef string c_fill_char = fill_char.encode("utf-8")

    with nogil:
        c_result = cpp_padding.pad(
            input.view(),
            width,
            side,
            c_fill_char,
        )

    return Column.from_libcudf(move(c_result))

cpdef Column zfill(Column input, size_type width):
    """
    Add '0' as padding to the left of each string.

    For details, see :cpp:func:`cudf::strings::zfill`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    width : int
        The minimum number of characters for each string.

    Returns
    -------
    Column
        New column of strings.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_padding.zfill(
            input.view(),
            width,
        )

    return Column.from_libcudf(move(c_result))
