from libcpp cimport bool
from libcpp.memory cimport unique_ptr

cimport cudf._libxx.lib as libcudf

cdef class _Column:
    cdef unique_ptr[libcudf.column] c_obj

    @staticmethod
    cdef from_ptr(unique_ptr[libcudf.column] ptr)

    cdef libcudf.size_type size(self)
    cdef libcudf.data_type type(self)
    cpdef bool has_nulls(self)


cdef class Column:
    cdef dict __dict__    
    cdef libcudf.column_view view(self)
    cdef libcudf.mutable_column_view mutable_view(self)
