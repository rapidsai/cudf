# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.pair cimport pair
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport bitmask_type, mask_state, size_type

from rmm.librmm.device_buffer cimport device_buffer
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/null_mask.hpp" namespace "cudf" nogil:
    cdef device_buffer copy_bitmask (
        column_view view,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef device_buffer copy_bitmask (
        const bitmask_type* null_mask,
        size_type begin_bit,
        size_type end_bit,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef size_t bitmask_allocation_size_bytes (
        size_type number_of_bits,
        size_t padding_boundary
    ) except +libcudf_exception_handler

    cdef device_buffer create_null_mask (
        size_type size,
        mask_state state,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef pair[device_buffer, size_type] bitmask_and(
        table_view view,
        cudaStream_t stream,
        device_async_resource_ref mr
    )

    cdef pair[device_buffer, size_type] bitmask_or(
        table_view view,
        cudaStream_t stream,
        device_async_resource_ref mr
    )

    cdef size_type null_count(
        const bitmask_type * bitmask,
        size_type start,
        size_type stop,
        cudaStream_t stream
    )

    cdef size_type index_of_first_set_bit(
        const bitmask_type * bitmask,
        size_type start,
        size_type stop,
        cudaStream_t stream
    )
