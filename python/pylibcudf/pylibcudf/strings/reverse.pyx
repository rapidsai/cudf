# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport reverse as cpp_reverse

__all__ = ["reverse"]

cpdef Column reverse(Column input):
    """Reverses the characters within each string.

    Any null string entries return corresponding null output column entries.

    For details, see :cpp:func:`reverse`.

    Parameters
    ----------
    input : Column
        Strings column for this operation

    Returns
    -------
    pylibcudf.Column
        New strings column
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_reverse.reverse(input.view())

    return Column.from_libcudf(move(c_result))
