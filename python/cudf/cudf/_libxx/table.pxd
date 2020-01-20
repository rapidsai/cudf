from libcpp.memory cimport unique_ptr

from cudf._libxx.lib cimport *
from cudf._libxx.column cimport *


cdef class Table:
    cdef dict __dict__

    cdef table_view view(self) except *
    cdef mutable_table_view mutable_view(self) except *
    cdef table_view data_view(self) except *
    cdef mutable_table_view mutable_data_view(self) except *
    cdef table_view index_view(self) except *
    cdef mutable_table_view mutable_index_view(self) except *

    @staticmethod
    cdef Table from_unique_ptr(unique_ptr[table] c_tbl, column_names,
                               index_names=*)
