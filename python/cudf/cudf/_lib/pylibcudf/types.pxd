# Copyright (c) 2023, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp cimport bool as cbool

from cudf._lib.cpp cimport types as cpp_types


cdef class DataType:
    cdef cpp_types.data_type c_obj

    cpdef cpp_types.type_id id(self)
    cpdef int32_t scale(self)

    @staticmethod
    cdef DataType from_libcudf(cpp_types.data_type dt)
