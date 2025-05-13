# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from libc.stddef cimport size_t
from libc.stdint cimport int32_t
from libcpp cimport bool as cbool
from pylibcudf.libcudf.types cimport (
    data_type,
    interpolation,
    mask_state,
    nan_equality,
    nan_policy,
    null_equality,
    null_order,
    null_policy,
    order,
    size_type,
    sorted,
    type_id,
)


cdef class DataType:
    cdef data_type c_obj

    cpdef type_id id(self)
    cpdef int32_t scale(self)

    @staticmethod
    cdef DataType from_libcudf(data_type dt)

cpdef size_t size_of(DataType t)
