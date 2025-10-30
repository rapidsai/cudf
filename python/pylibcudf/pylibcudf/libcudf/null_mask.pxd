# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.pair cimport pair
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport bitmask_type, mask_state, size_type

from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/null_mask.hpp" namespace "cudf" nogil:
    cdef device_buffer copy_bitmask "cudf::copy_bitmask" (
        column_view view,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef size_t bitmask_allocation_size_bytes (
        size_type number_of_bits,
        size_t padding_boundary
    ) except +libcudf_exception_handler

    cdef device_buffer create_null_mask (
        size_type size,
        mask_state state,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef pair[device_buffer, size_type] bitmask_and(
        table_view view,
        cuda_stream_view stream,
        device_memory_resource* mr
    )

    cdef pair[device_buffer, size_type] bitmask_or(
        table_view view,
        cuda_stream_view stream,
        device_memory_resource* mr
    )

    cdef size_type null_count(
        const bitmask_type * bitmask,
        size_type start,
        size_type stop,
        cuda_stream_view stream
    )
