from cudf._lib.cudf cimport *

cdef class Table:
    cdef cudf_table* ptr
    cdef vector[gdf_column*] c_columns
