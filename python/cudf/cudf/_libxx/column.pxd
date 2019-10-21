cimport cudf._libxx.lib as libcuxx

cdef class Column:
    cdef libcuxx.column *c_obj

cdef class ColumnView:
    cdef Column owner
    cdef libcuxx.column_view *c_obj

cdef class MutableColumnView:
    cdef Column owner
    cdef libcuxx.mutable_column_view *c_obj
