# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.nvtext.dedup cimport (
    substring_deduplicate as cpp_substring_deduplicate,
)
from pylibcudf.libcudf.types cimport size_type

__all__ = ["substring_deduplicate"]


cpdef Column substring_deduplicate(Column input, size_type min_width):
    """
    Returns duplicate strings found anywhere in the input column
    with min_width minimum number of bytes.

    For details, see :cpp:func:`substring_deduplicate`

    Parameters
    ----------
    input : Column
        Strings column of text
    min_width : size_type
        Minimum width of bytes to detect duplicates

    Returns
    -------
    Column
        New column of duplicate strings
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_substring_deduplicate(input.view(), min_width)

    return Column.from_libcudf(move(c_result))
