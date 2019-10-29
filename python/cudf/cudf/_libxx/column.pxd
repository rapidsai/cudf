from libcpp cimport bool
from libcpp.memory cimport unique_ptr

cimport cudf._libxx.lib as libcudf

cdef class _Column:
    cdef unique_ptr[libcudf.column] c_obj

    @staticmethod
    cdef _Column from_ptr(unique_ptr[libcudf.column] ptr)

    cdef libcudf.size_type size(self) except *
    cdef libcudf.data_type type(self) except *
    cpdef bool has_nulls(self) except *


cdef class Column:
    cdef dict __dict__    
    cdef libcudf.column_view view(self) except *
    cdef libcudf.mutable_column_view mutable_view(self) except *
