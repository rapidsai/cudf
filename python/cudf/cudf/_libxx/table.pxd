from libcpp.memory cimport unique_ptr

from cudf._libxx.lib cimport *
from cudf._libxx.column cimport *


cdef class _Table:
    cdef unique_ptr[table] c_obj

    @staticmethod
    cdef _Table from_ptr(unique_ptr[table] ptr)


cdef class Table:
    cdef dict __dict__
    cdef table_view view(self)
    cdef mutable_table_view mutable_view(self)
