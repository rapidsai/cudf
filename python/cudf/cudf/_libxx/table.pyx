from collections import OrderedDict

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
                "{value.__class__.__name__} into OrderedColumnDict"
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


cdef class _Table:

    def __init__(self, data=None, names=None):
        if data is None:
            data = OrderedColumnDict()
        if isinstance(data, OrderedColumnDict):
            self._data = data
        else:
            if names is None:
                names = range(len(data))
            self._data = OrderedColumnDict(zip(names, data))

    cdef table_view view(self) except *:
        cdef vector[column_view] column_views

        cdef Column col
        for col in self._data.values():
            column_views.push_back(col.view())

        return table_view(column_views)

    cdef mutable_table_view mutable_view(self) except *:
        cdef vector[mutable_column_view] column_views

        cdef Column col
        for col in self._data.values():
            column_views.push_back(col.mutable_view())

        return mutable_table_view(column_views)

    @staticmethod
    cdef _Table from_ptr(unique_ptr[table] c_tbl, names=None):
        cdef vector[unique_ptr[column]] columns
        columns = c_tbl.get()[0].release()
        result = []
        for i in range(columns.size()):
            result.append(Column.from_ptr(move(columns[i])))
        return _Table(result)
