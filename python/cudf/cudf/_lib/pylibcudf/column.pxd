# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column


cdef class Column:
    cdef unique_ptr[column] c_obj
    cdef column * get(self) nogil
