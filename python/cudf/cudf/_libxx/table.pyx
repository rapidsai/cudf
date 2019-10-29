import numpy as np

from libc.stdint cimport uintptr_t

from cudf._libxx.column cimport *
from cudf._libxx.lib cimport *

cdef class _Table:

    def __cinit__(self):
        pass

    @staticmethod
    cdef _Table from_ptr(unique_ptr[table] ptr):
        cdef _Table tbl = _Table.__new__(_Table)
        tbl.c_obj = move(ptr)
        return tbl

    def release_into_table(self):
        cdef vector[unique_ptr[column]] columns
        columns = self.c_obj.get()[0].release()
        result = []
        for i in range(columns.size()):
            result.append(_Column.from_ptr(move(columns[i])).release_into_column())
        return result
        

cdef class Table:

    def __init__(self, columns):
        self.columns = columns

    cdef table_view view(self) except *:
        cdef vector[column_view] column_views

        cdef Column col
        for i in range(len(self.columns)):
            col = self.columns[i]
            column_views.push_back(col.view())
        
        return table_view(column_views)

    cdef mutable_table_view mutable_view(self) except *:
        cdef vector[mutable_column_view] column_views

        cdef Column col
        for i in range(len(self.columns)):
            col = self.columns[i]
            column_views.push_back(col.mutable_view())
        
        return mutable_table_view(column_views)

