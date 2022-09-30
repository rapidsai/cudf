# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libc.stdint cimport int32_t

from rmm._lib.device_buffer cimport device_buffer

cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib.cpp.column.column_view cimport column_view

ctypedef int32_t underlying_type_t_mask_state


cdef extern from "cudf/null_mask.hpp" namespace "cudf" nogil:
    cdef device_buffer copy_bitmask "cudf::copy_bitmask" (
        column_view view
    ) except +

    cdef size_t bitmask_allocation_size_bytes (
        libcudf_types.size_type number_of_bits,
        size_t padding_boundary
    ) except +

    cdef size_t bitmask_allocation_size_bytes (
        libcudf_types.size_type number_of_bits
    ) except +

    cdef device_buffer create_null_mask (
        libcudf_types.size_type size,
        libcudf_types.mask_state state
    ) except +
