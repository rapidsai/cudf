# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from libc.stdint cimport int32_t

from cudf._lib.cpp.types cimport data_type, type_id


cdef type_id py_type_to_c_type(py_type_id)

cdef class DataType:
    cdef data_type c_obj
    cpdef id(self)
    cpdef int32_t scale(self)
