from cudf._lib.cudf cimport *

cdef class Table:
    """
    Python wrapper around cudf::table
    """
    def __cinit__(self, columns=None):
        cdef gdf_column* c_col
        if columns is not None:
            for col in columns:
                c_col = column_view_from_column(col)
                self.c_columns.push_back(c_col)
            self.ptr = new cudf_table(self.c_columns)
        else:
            self.ptr = new cudf_table()

    def __init__(self, columns=None):
        pass

    def get_columns(self, own):
        cols = []
        cdef i
        cdef gdf_column* c_col
        for i in range(self.num_columns):
            c_col = self.ptr[0].get_column(i)
            col = gdf_column_to_column(c_col, False, own=own)
            cols.append(col)
        return cols

    @property
    def num_columns(self):
        return self.ptr[0].num_columns()

    @property
    def num_rows(self):
        return self.ptr[0].num_rows()
    
    def __dealloc__(self):
        cdef i
        for i in range(self.ptr[0].num_columns()):
            free_column(self.ptr[0].get_column(i))
        del self.ptr


