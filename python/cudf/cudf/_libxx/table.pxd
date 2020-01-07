from libcpp.memory cimport unique_ptr

from cudf._libxx.lib cimport *
from cudf._libxx.column cimport *

cdef class _Table:
    cdef dict __dict__
    cdef table_view view(self) except *
    cdef mutable_table_view mutable_view(self) except *

    @staticmethod
    cdef _Table from_ptr(unique_ptr[table] c_tbl, names=*)
