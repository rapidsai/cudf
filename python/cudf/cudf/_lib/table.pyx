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


cdef class Table:
    def __init__(self, object data=None, object index=None):
        """
        Table: A collection of Column objects with an optional index.

        Parameters
        ----------
        data : dict
            An dict mapping column names to Columns
        index : Table
            A Table representing the (optional) index columns.
        """
        if data is None:
            data = {}
        self._data = cudf.core.column_accessor.ColumnAccessor(data)
        self._index = index

    @property
    def _num_columns(self):
        return len(self._data)

    @property
    def _num_indices(self):
        if self._index is None:
            return 0
        else:
            return len(self._index_names)

    @property
    def _num_rows(self):
        if self._index is not None:
            return len(self._index)
        if len(self._data) == 0:
            return 0
        return len(self._data.columns[0])

    @property
    def _column_names(self):
        return self._data.names

    @property
    def _index_names(self):
        return None if self._index is None else self._index._data.names

    @property
    def _columns(self):
        """
        Return a list of Column objects backing this dataframe
        """
        return self._data.columns

    cdef table_view view(self) except *:
        """
        Return a cudf::table_view of all columns (including index columns)
        of this Table.
        """
        if self._index is None:
            return make_table_view(
                self._data.columns
            )
        return make_table_view(
            itertools.chain(
                self._index._data.columns,
                self._data.columns,
            )
        )

    cdef table_view data_view(self) except *:
        """
        Return a cudf::table_view of just the data columns
        of this Table.
        """
        return make_table_view(
            self._data.columns
        )


cdef table_view make_table_view(columns) except*:
    """
    Helper function to create a cudf::table_view from
    a list of Columns
    """
    cdef vector[column_view] column_views

    cdef Column col
    for col in columns:
        column_views.push_back(col.view())

    return table_view(column_views)
