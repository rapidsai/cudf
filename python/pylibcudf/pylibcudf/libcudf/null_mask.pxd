# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp.pair cimport pair
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport bitmask_type, mask_state, size_type

from rmm.librmm.device_buffer cimport device_buffer


cdef extern from "cudf/null_mask.hpp" namespace "cudf" nogil:
    cdef device_buffer copy_bitmask "cudf::copy_bitmask" (
        column_view view
    ) except +libcudf_exception_handler

    cdef size_t bitmask_allocation_size_bytes (
        size_type number_of_bits,
        size_t padding_boundary
    ) except +libcudf_exception_handler

    cdef size_t bitmask_allocation_size_bytes (
        size_type number_of_bits
    ) except +libcudf_exception_handler

    cdef device_buffer create_null_mask (
        size_type size,
        mask_state state
    ) except +libcudf_exception_handler

    cdef pair[device_buffer, size_type] bitmask_and(
        table_view view
    )

    cdef pair[device_buffer, size_type] bitmask_or(
        table_view view
    )

    cdef size_type null_count(
        const bitmask_type * bitmask,
        size_type start,
        size_type stop,
    )
