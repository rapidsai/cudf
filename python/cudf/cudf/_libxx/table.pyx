from collections import OrderedDict

import numpy as np

from libc.stdint cimport uintptr_t

from cudf._libxx.column cimport *
from cudf._libxx.lib cimport *


cdef class _Table:

    def __init__(self, columns):
        """
        Data: an iterable of Columns
        """
        self._columns = columns

    cdef table_view view(self) except *:
        cdef vector[column_view] column_views

        cdef Column col
        for col in self._columns:
            column_views.push_back(col.view())

        return table_view(column_views)

    cdef mutable_table_view mutable_view(self) except *:
        cdef vector[mutable_column_view] column_views

        cdef Column col
        for col in self._columns:
            column_views.push_back(col.mutable_view())

        return mutable_table_view(column_views)

    @staticmethod
    cdef _Table from_unique_ptr(unique_ptr[table] c_tbl):
        cdef vector[unique_ptr[column]] columns
        columns = c_tbl.get()[0].release()
        result = []
        for i in range(columns.size()):
            result.append(Column.from_unique_ptr(move(columns[i])))
        return _Table(result)
