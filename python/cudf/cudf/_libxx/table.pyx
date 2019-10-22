import numpy as np

from libc.stdint cimport uintptr_t

from cudf._libxx.column cimport *
from cudf._libxx.lib cimport *


cdef class TableView:
    def __cinit__(self):
        pass

    @classmethod
    def from_column_views(cls, column_views):
        cdef TableView tview = TableView.__new__(TableView)
        cdef vector[column_view] cols
        cdef ColumnView col
        for i in range(len(column_views)):
            col = column_views[i]
            cols.push_back(col.c_obj[0])
        tview.c_obj = new table_view(cols)
        return tview

    def __dealloc__(self):
        del self.c_obj
