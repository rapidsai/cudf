# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int32_t

from rmm._lib.device_buffer cimport device_buffer

from cudf._libxx.cpp.column.column_view cimport column_view
cimport cudf._libxx.cpp.types as cudf_types

ctypedef int32_t mask_state_underlying_type


cdef extern from "cudf/null_mask.hpp" namespace "cudf" nogil:
    cdef device_buffer copy_bitmask "cudf::copy_bitmask" (
        column_view view
    ) except +

    cdef size_t bitmask_allocation_size_bytes (
        cudf_types.size_type number_of_bits,
        size_t padding_boundary
    ) except +

    cdef size_t bitmask_allocation_size_bytes (
        cudf_types.size_type number_of_bits
    ) except +

    cdef device_buffer create_null_mask (
        cudf_types.size_type size,
        cudf_types.mask_state state
    ) except +
