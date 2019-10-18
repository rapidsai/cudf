cimport cudf._libxx.lib as libcuxx

cdef class Column:
    cdef libcuxx.column *c_obj

cdef class ColumnView:
    cdef libcuxx.column_view *c_obj
