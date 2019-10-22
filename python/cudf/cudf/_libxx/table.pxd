from cudf._libxx.lib cimport *

cdef class Table:
    cdef table *c_obj

cdef class TableView:
    cdef Table owner
    cdef table_view *c_obj

cdef class MutableTableView:
    cdef Table owner
    cdef mutable_table_view *c_obj
