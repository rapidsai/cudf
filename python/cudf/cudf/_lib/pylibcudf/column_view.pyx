# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.vector cimport vector

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport bitmask_type, data_type, size_type, type_id


cdef class ColumnView:
    """Wrapper around column_view."""
    cdef column_view * thisptr

    def __cinit__(self, size_type size):
        cdef data_type dtype = data_type(type_id.INT32)
        cdef const void * data = NULL
        cdef const bitmask_type * null_mask = NULL
        cdef size_type null_count = 0
        cdef size_type offset = 0
        cdef const vector[column_view] children

        self.thisptr = new column_view(
            dtype, size, data, null_mask, null_count, offset, children
        )

    def __dealloc__(self):
        del self.thisptr
