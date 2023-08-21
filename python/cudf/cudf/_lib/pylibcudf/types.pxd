# Copyright (c) 2023, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp cimport bool as cbool

from cudf._lib.cpp.types cimport data_type, type_id


cdef class DataType:
    cdef data_type c_obj

    cpdef type_id id(self)
    cpdef int32_t scale(self)

    @staticmethod
    cdef DataType from_libcudf(data_type dt)
