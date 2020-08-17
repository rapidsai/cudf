# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
from cudf._lib.table cimport Table
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from libcpp.memory cimport unique_ptr, shared_ptr
from cudf._lib.move cimport move

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from pyarrow.lib cimport CTable, pyarrow_wrap_table, pyarrow_unwrap_table
from cudf._lib.cpp.interop cimport (
    to_arrow as cpp_to_arrow,
    from_arrow as cpp_from_arrow
)


def to_arrow(Table input_table, object column_names, bool keep_index=True):
    """Convert from a cudf Table to PyArrow Table.

    Parameters
    ----------
    input_table : cudf table
    column_names : names for the pyarrow arrays
    keep_index : whether index needs to be part of arrow table

    Returns
    -------
    pyarrow table
    """

    cdef vector[string] cpp_column_names
    cdef table_view input = (
        input_table.view() if keep_index else input_table.data_view()
    )
    cpp_column_names.reserve(len(column_names))
    for name in column_names:
        cpp_column_names.push_back(str.encode(str(name)))

    cdef shared_ptr[CTable] cpp_arrow_table
    with nogil:
        cpp_arrow_table = cpp_to_arrow(input, cpp_column_names)

    return pyarrow_wrap_table(cpp_arrow_table)


def from_arrow(
    object input_table,
    object column_names=None,
    object index_names=None
):
    """Convert from a PyArrow Table to cudf Table.

    Parameters
    ----------
    input_table : PyArrow table
    column_names : names for the cudf table data columns
    index_names : names for the cudf table index columns

    Returns
    -------
    cudf Table
    """
    cdef shared_ptr[CTable] cpp_arrow_table = (
        pyarrow_unwrap_table(input_table)
    )
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_from_arrow(cpp_arrow_table.get()[0]))

    out_table = Table.from_unique_ptr(
        move(c_result),
        column_names=column_names,
        index_names=index_names
    )

    return out_table
