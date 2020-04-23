# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.legacy.cudf cimport *
from libcpp.memory cimport unique_ptr, make_unique

cdef class TableView:
    """
    A non-owning view into a set of Columns
    """
    def __cinit__(self, columns):
        cdef gdf_column* c_col
        for col in columns:
            c_col = column_view_from_column(col)
            self.c_columns.push_back(c_col)
        self.ptr = new cudf_table(self.c_columns)

    def __init__(self, columns):
        pass

    def __dealloc__(self):
        del self.ptr


cdef class Table:
    """
    A set of Columns. Wraps cudf::table.
    """
    def __cinit__(self):
        self.ptr = make_unique[cudf_table]()

    def __init__(self):
        pass

    @staticmethod
    cdef from_ptr(unique_ptr[cudf_table]&& ptr):
        cdef Table tbl = Table.__new__(Table)
        tbl.ptr = move(ptr)
        return tbl

    def release(self):
        """
        Releases ownership of the Columns and returns
        them as a list. After `release()` is called,
        the Table is empty.
        """
        cols = []
        cdef i
        cdef gdf_column* c_col
        for i in range(self.num_columns):
            c_col = self.ptr.get().get_column(i)
            col = gdf_column_to_column(c_col)
            cols.append(col)
        self.ptr = make_unique[cudf_table]()
        return cols

    @property
    def num_columns(self):
        if self.ptr.get() is not NULL:
            return self.ptr.get().num_columns()
        return 0

    @property
    def num_rows(self):
        if self.ptr.get() is not NULL:
            return self.ptr.get().num_rows()
        return 0

    def __dealloc__(self):
        cdef i
        for i in range(self.ptr.get().num_columns()):
            free_column(self.ptr.get().get_column(i))
