# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp.pair cimport pair

from rmm._lib.device_buffer cimport device_buffer

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport mask_state, size_type

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

    cdef pair[device_buffer, size_type] bitmask_and(
        table_view view
    )

    cdef pair[device_buffer, size_type] bitmask_or(
        table_view view
    )
