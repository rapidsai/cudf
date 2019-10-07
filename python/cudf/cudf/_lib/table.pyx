from cudf._lib.cudf cimport *


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
        cdef i
        for i in range(self.ptr[0].num_columns()):
            free_column(self.ptr[0].get_column(i))
        del self.ptr


cdef class Table:
    """
    A set of Columns. Wraps cudf::table.
    """
    def __cinit__(self):
        self.ptr = new cudf_table()
    
    def __init__(self):
        pass

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
            c_col = self.ptr[0].get_column(i)
            col = gdf_column_to_column(c_col)
            cols.append(col)
        self.ptr = new cudf_table()
        return cols

    @property
    def num_columns(self):
        if self.ptr is not NULL:
            return self.ptr[0].num_columns()
        return 0

    @property
    def num_rows(self):
        if self.ptr is not NULL:
            return self.ptr[0].num_rows()
        return 0

    def __dealloc__(self):
        cdef i
        for i in range(self.ptr[0].num_columns()):
            free_column(self.ptr[0].get_column(i))
        del self.ptr
