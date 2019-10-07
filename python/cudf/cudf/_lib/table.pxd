from cudf._lib.cudf cimport *

cdef class TableView:
    cdef cudf_table* ptr
    cdef vector[gdf_column*] c_columns

cdef class Table:
    cdef cudf_table* ptr
    cdef vector[gdf_column*] c_columns

    @staticmethod
    cdef Table from_ptr(cudf_table* ptr)
