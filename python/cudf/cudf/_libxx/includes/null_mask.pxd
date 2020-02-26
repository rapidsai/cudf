# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int32_t

from cudf._libxx.lib cimport *


ctypedef int32_t underlying_type_t_mask_state

cdef extern from "cudf/null_mask.hpp" namespace "cudf" nogil:
    cdef device_buffer copy_bitmask "cudf::copy_bitmask" (
        column_view view
    ) except +

    cdef size_t bitmask_allocation_size_bytes (
        size_type number_of_bits,
        size_t padding_boundary
    ) except +

    cdef size_t bitmask_allocation_size_bytes (
        size_type number_of_bits
    ) except +

    cdef device_buffer create_null_mask (
        size_type size,
        mask_state state
    ) except +
