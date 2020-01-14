from collections import OrderedDict
import itertools

import numpy as np

from libc.stdint cimport uintptr_t

from cudf._libxx.column cimport *
from cudf._libxx.lib cimport *


class OrderedColumnDict(OrderedDict):
    def __setitem__(self, key, value):
        from cudf.core.column import ColumnBase

        if not isinstance(value, ColumnBase):
            raise TypeError(
                f"Cannot insert object of type "
                f"{value.__class__.__name__} into OrderedColumnDict"
            )

        if self.first is not None and len(self.first) > 0:
            if len(value) != len(self.first):
                raise ValueError(
                    f"Cannot insert Column of different length "
                    "into OrderedColumnDict"
                )

        super().__setitem__(key, value)

    @property
    def first(self):
        """
        Returns the first value if self is non-empty;
        returns None otherwise.
        """
        if len(self) == 0:
            return None
        else:
            return next(iter(self.values()))


cdef class Table:

    def __init__(self, data=None, index=None):
        """
        Data: an iterable of Columns
        """
        if data is None:
            data = OrderedColumnDict({})
        self._data = data
        self._index = index

    @property
    def _column_names(self):
        return self._data.keys()

    @property
    def _index_names(self):
        if self._index is not None:
            return self._index._column_names
        return None


    cdef table_view view(self) except *:
        return self._make_table_view(self._data.values())

    cdef mutable_table_view mutable_view(self) except *:
        return self._make_mutable_table_view(self._data.values())

    cdef table_view indexed_view(self) except *:
        if self._index is None:
            return self.view()
        return self._make_table_view(
            itertools.chain(
                self._data.values(),
                self._index._data.values()
            )
        )

    cdef mutable_table_view mutable_indexed_view(self) except *:
        if self._index is None:
            return self.mutable_view()
        return self._make_mutable_table_view(
            itertools.chain(
                self._data.values(),
                self._index._data.values()
            )
        )
    
    cdef table_view _make_table_view(self, columns) except*:
        cdef vector[column_view] column_views

        cdef Column col
        for col in columns:
            column_views.push_back(col.view())

        return table_view(column_views)

    cdef mutable_table_view _make_mutable_table_view(self, columns) except*:
        cdef vector[mutable_column_view] mutable_column_views

        cdef Column col
        for col in columns:
            mutable_column_views.push_back(col.mutable_view())

        return mutable_table_view(mutable_column_views)
    
    @staticmethod
    cdef Table from_unique_ptr(unique_ptr[table] c_tbl, column_names=None,
                               index_names=None):
        cdef vector[unique_ptr[column]] columns
        columns = c_tbl.get()[0].release()

        index = None
        num_index_columns = 0
        
        if index_names:
            num_index_columns = len(index_names)
            index_columns = []
            for i in range(len(index_names)):
                index_columns.append(Column.from_unique_ptr(move(columns[i])))
            index = Table(OrderedColumnDict(zip(index_names, index_columns)))

        data_columns = []
        for i in range(num_index_columns, columns.size()):
            data_columns.append(Column.from_unique_ptr(move(columns[i])))
        data = OrderedColumnDict(zip(column_names, data_columns))

        return Table(data=data, index=index)
