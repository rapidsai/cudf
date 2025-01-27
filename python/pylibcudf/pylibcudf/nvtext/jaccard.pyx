# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.jaccard cimport (
    jaccard_index as cpp_jaccard_index,
)
from pylibcudf.libcudf.types cimport size_type

__all__ = ["jaccard_index"]

cpdef Column jaccard_index(Column input1, Column input2, size_type width):
    """
    Returns the Jaccard similarity between individual rows in two strings columns.

    For details, see :cpp:func:`jaccard_index`

    Parameters
    ----------
    input1 : Column
        Input strings column
    input2 : Column
        Input strings column
    width : size_type
        The ngram number to generate

    Returns
    -------
    Column
        Index calculation values
    """
    cdef column_view c_input1 = input1.view()
    cdef column_view c_input2 = input2.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_jaccard_index(
            c_input1,
            c_input2,
            width
        )

    return Column.from_libcudf(move(c_result))
