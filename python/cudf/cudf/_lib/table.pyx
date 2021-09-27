# Copyright (c) 2020, NVIDIA CORPORATION.

import itertools

import numpy as np

from cython.operator cimport dereference
from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view, mutable_column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport mutable_table_view, table_view
from cudf._lib.cpp.types cimport size_type

import cudf


cdef table_view table_view_from_columns(columns) except*:
    """Create a cudf::table_view from an iterable of Columns."""
    cdef vector[column_view] column_views

    cdef Column col
    for col in columns:
        column_views.push_back(col.view())

    return table_view(column_views)


cdef table_view table_view_from_table(tbl, ignore_index=False) except*:
    """Create a cudf::table_view from a Table.

    Parameters
    ----------
    ignore_index : bool, default False
        If True, don't include the index in the columns.
    """
    return table_view_from_columns(
        tbl._index._data.columns + tbl._data.columns
        if not ignore_index and tbl._index is not None
        else tbl._data.columns
    )
