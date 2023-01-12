# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type


cdef class TableView:
    cdef unique_ptr[table_view] c_obj
    cdef list columns
    cdef table_view * get(self) nogil
    cpdef size_type num_columns(self)
    cpdef size_type num_rows(self)
