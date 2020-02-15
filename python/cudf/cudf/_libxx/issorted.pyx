from cudf._libxx.cudf cimport *
from cudf._libxx.cudf import *
from cudf._libxx.utils cimport *
from cudf._libxx.utils import *

from cudf._libxx.includes.issorted cimport is_sorted

def issorted(Table source_table, order=[], null_prec=[]):
    """
    Checks whether the rows of a `table` are sorted in lexicographical order.

    Parameters
    ----------
    source_table : table whose columns are to be
                   checked for sort order
    order : expected sort order of each column
            size must be len(columns) or empty
            if empty, all columns expected sort
            order is set to ascending
    null_prec : desired order of null
                compared to other elements for each column
                size must be len(columns) or empty
                if empty, null order is set to before

    Returns
    -------
    returns : boolean
              true, if sorted as expected in order
              false, otherwise
    """

    if 0 < len(order):
        cdef vector[order] column_order = vector[order](len(order), order.ASCENDING)
    else:
        cdef vector[order] column_order
    if 0 < len(null_order):
        cdef vector[null_order] null_precedence = vector[null_order](len(null_prec), null_order.BEFORE)
    else:
        cdef vector[null_order] null_precedence
    return is_sorted(source_table.view(),
                     vector[order] column_order,
                     vector[null_order] null_precedence)
