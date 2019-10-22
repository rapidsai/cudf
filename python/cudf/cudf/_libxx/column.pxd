cimport cudf._libxx.lib as libcudf

cdef class Column:
    cdef libcudf.column *c_obj

cdef class ColumnView:
    cdef Column owner
    cdef libcudf.column_view *c_obj

cdef class MutableColumnView:
    cdef Column owner
    cdef libcudf.mutable_column_view *c_obj
