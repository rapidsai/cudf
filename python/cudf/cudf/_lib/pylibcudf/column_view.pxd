# Copyright (c) 2022, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport bitmask_type, data_type, size_type, type_id

from .types cimport py_type_to_c_type
from .utils cimport int_to_bitmask_ptr, int_to_void_ptr


cdef class ColumnView:
    cdef unique_ptr[column_view] c_obj
    cdef column_view get(self) nogil


cdef class Column:
    cdef unique_ptr[column] c_obj
