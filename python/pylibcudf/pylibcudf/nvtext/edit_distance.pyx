# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.edit_distance cimport (
    edit_distance as cpp_edit_distance,
    edit_distance_matrix as cpp_edit_distance_matrix,
)

__all__ = ["edit_distance", "edit_distance_matrix"]

cpdef Column edit_distance(Column input, Column targets):
    """
    Returns the edit distance between individual strings in two strings columns

    For details, see :cpp:func:`edit_distance`

    Parameters
    ----------
    input : Column
        Input strings
    targets : Column
        Strings to compute edit distance against

    Returns
    -------
    Column
        New column of edit distance values
    """
    cdef column_view c_strings = input.view()
    cdef column_view c_targets = targets.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_edit_distance(c_strings, c_targets)

    return Column.from_libcudf(move(c_result))


cpdef Column edit_distance_matrix(Column input):
    """
    Returns the edit distance between all strings in the input strings column

    For details, see :cpp:func:`edit_distance_matrix`

    Parameters
    ----------
    input : Column
        Input strings

    Returns
    -------
    Column
        New column of edit distance values
    """
    cdef column_view c_strings = input.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_edit_distance_matrix(c_strings)

    return Column.from_libcudf(move(c_result))
