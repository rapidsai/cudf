
from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cudf._libxx.lib cimport *

cdef class Column:
    cdef dict __dict__    
    cdef column_view view(self) except *
    cdef mutable_column_view mutable_view(self) except *

    @staticmethod
    cdef Column from_ptr(unique_ptr[column] c_col)

    cdef size_type null_count(self)
