# Copyright (c) 2020, NVIDIA CORPORATION.

import itertools

import numpy as np

from cudf.core.column_accessor import ColumnAccessor

from cython.operator cimport dereference
from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column

from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport (
    column_view,
    mutable_column_view
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport (
    table_view,
    mutable_table_view
)


cdef class Table:
    def __init__(self, object data=None, object index=None):
        """
        Table: A collection of Column objects with an optional index.

        Parameters
        ----------
        data : OrderedColumnDict
            An OrderedColumnDict mapping column names to Columns
        index : Table
            A Table representing the (optional) index columns.
        """
        if data is None:
            data = {}
        self._data = ColumnAccessor(data)
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

    @staticmethod
    cdef Table from_unique_ptr(
        unique_ptr[table] c_tbl,
        object column_names,
        object index_names=None
    ):
        """
        Construct a Table from a unique_ptr to a cudf::table.

        Parameters
        ----------
        c_tbl : unique_ptr[cudf::table]
        index_names : iterable
        column_names : iterable
        """
        cdef vector[unique_ptr[column]] columns
        columns = move(c_tbl.get()[0].release())

        cdef vector[unique_ptr[column]].iterator it = columns.begin()

        # First construct the index, if any
        index = None
        if index_names is not None:
            index_columns = []
            for _ in index_names:
                index_columns.append(Column.from_unique_ptr(
                    move(dereference(it))
                ))
                it += 1
            index = Table(dict(zip(index_names, index_columns)))

        # Construct the data OrderedColumnDict
        data_columns = []
        for _ in column_names:
            data_columns.append(Column.from_unique_ptr(move(dereference(it))))
            it += 1
        data = dict(zip(column_names, data_columns))

        return Table(data=data, index=index)

    @staticmethod
    cdef Table from_table_view(
        table_view tv,
        object owner,
        object column_names,
        object index_names=None
    ):
        """
        Given a ``cudf::table_view``, constructs a ``cudf.Table`` from it,
        along with referencing an ``owner`` Python object that owns the memory
        lifetime. If ``owner`` is a ``cudf.Table``, we reach inside of it and
        reach inside of each ``cudf.Column`` to make the owner of each newly
        created ``Buffer`` underneath the ``cudf.Column`` objects of the
        created ``cudf.Table`` the respective ``Buffer`` from the relevant
        ``cudf.Column`` of the ``owner`` ``cudf.Table``.
        """
        cdef size_type column_idx = 0
        table_owner = isinstance(owner, Table)

        # First construct the index, if any
        index = None
        if index_names is not None:
            index_columns = []
            for _ in index_names:
                column_owner = owner
                if table_owner:
                    column_owner = owner._index._columns[column_idx]
                index_columns.append(
                    Column.from_column_view(
                        tv.column(column_idx),
                        column_owner
                    )
                )
                column_idx += 1
            index = Table(dict(zip(index_names, index_columns)))

        # Construct the data OrderedColumnDict
        cdef size_type source_column_idx = 0
        data_columns = []
        for _ in column_names:
            column_owner = owner
            if table_owner:
                column_owner = owner._columns[source_column_idx]
            data_columns.append(
                Column.from_column_view(tv.column(column_idx), column_owner)
            )
            column_idx += 1
            source_column_idx += 1
        data = dict(zip(column_names, data_columns))

        return Table(data=data, index=index)

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

    cdef mutable_table_view mutable_view(self) except *:
        """
        Return a cudf::mutable_table_view of all columns
        (including index columns) of this Table.
        """
        if self._index is None:
            return make_mutable_table_view(
                self._data.columns
            )
        return make_mutable_table_view(
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

    cdef mutable_table_view mutable_data_view(self) except *:
        """
        Return a cudf::mutable_table_view of just the data columns
        of this Table.
        """
        return make_mutable_table_view(
            self._data.columns
        )

    cdef table_view index_view(self) except *:
        """
        Return a cudf::table_view of just the index columns
        of this Table.
        """
        if self._index is None:
            raise ValueError("Cannot get index_view of a Table "
                             "that has no index")
        return make_table_view(
            self._index.values()
        )

    cdef mutable_table_view mutable_index_view(self) except *:
        """
        Return a cudf::mutable_table_view of just the index columns
        of this Table.
        """
        if self._index is None:
            raise ValueError("Cannot get mutable_index_view of a Table "
                             "that has no index")
        return make_mutable_table_view(
            self._index._data.columns
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

cdef mutable_table_view make_mutable_table_view(columns) except*:
    """
    Helper function to create a cudf::mutable_table_view from
    a list of Columns
    """
    cdef vector[mutable_column_view] mutable_column_views

    cdef Column col
    for col in columns:
        mutable_column_views.push_back(col.mutable_view())

    return mutable_table_view(mutable_column_views)

cdef columns_from_ptr(unique_ptr[table] c_tbl):
    """
    Return a list of table columns from a unique pointer

    Parameters
    ----------
    c_tbl : unique_ptr[cudf::table]
    """
    num_columns = c_tbl.get().num_columns()
    cdef vector[unique_ptr[column]] columns
    columns = move(c_tbl.get()[0].release())
    cdef vector[unique_ptr[column]].iterator it = columns.begin()

    result = [None] * num_columns
    for i in range(num_columns):
        result[i] = Column.from_unique_ptr(move(dereference(it)))
        it += 1
    return result
