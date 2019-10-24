cimport cudf._libxx.lib as libcudf
from libcpp.memory cimport unique_ptr


cdef class _Column:
    cdef unique_ptr[libcudf.column] owner


cdef class Column:
    cdef libcudf.column_view view(self)
    cdef libcudf.mutable_column_view mutable_view(self)
