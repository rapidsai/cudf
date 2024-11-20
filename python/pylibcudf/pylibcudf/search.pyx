# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf cimport search as cpp_search
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.types cimport null_order, order

from .column cimport Column
from .table cimport Table

__all__ = ["contains", "lower_bound", "upper_bound"]

cpdef Column lower_bound(
    Table haystack,
    Table needles,
    list column_order,
    list null_precedence,
):
    """Find smallest indices in haystack where needles may be inserted to retain order.

    For details, see :cpp:func:`lower_bound`.

    Parameters
    ----------
    haystack : Table
        The search space.
    needles : Table
        The values for which to find insertion points.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Column
        The insertion points
    """
    cdef unique_ptr[column] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = cpp_search.lower_bound(
            haystack.view(),
            needles.view(),
            c_orders,
            c_null_precedence,
        )
    return Column.from_libcudf(move(c_result))


cpdef Column upper_bound(
    Table haystack,
    Table needles,
    list column_order,
    list null_precedence,
):
    """Find largest indices in haystack where needles may be inserted to retain order.

    For details, see :cpp:func:`upper_bound`.

    Parameters
    ----------
    haystack : Table
        The search space.
    needles : Table
        The values for which to find insertion points.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Column
        The insertion points
    """
    cdef unique_ptr[column] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence
    with nogil:
        c_result = cpp_search.upper_bound(
            haystack.view(),
            needles.view(),
            c_orders,
            c_null_precedence,
        )
    return Column.from_libcudf(move(c_result))


cpdef Column contains(Column haystack, Column needles):
    """Check whether needles are present in haystack.

    For details, see :cpp:func:`contains`.

    Parameters
    ----------
    haystack : Table
        The search space.
    needles : Table
        The values for which to search.

    Returns
    -------
    Column
        Boolean indicator for each needle.
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_search.contains(
            haystack.view(),
            needles.view(),
        )
    return Column.from_libcudf(move(c_result))
