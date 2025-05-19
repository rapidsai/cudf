# Copyright (c) 2023-2025, NVIDIA CORPORATION.
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport device_span
from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/contiguous_split.hpp" namespace "cudf" nogil:
    cdef cppclass packed_columns:
        unique_ptr[vector[uint8_t]] metadata
        unique_ptr[device_buffer] gpu_data

    cdef cppclass chunked_pack:
        bool has_next()
        size_t get_total_contiguous_size()
        size_t next(
            const device_span[uint8_t] &
        ) except +libcudf_exception_handler
        unique_ptr[vector[uint8_t]] build_metadata(
        ) except +libcudf_exception_handler

        @staticmethod
        unique_ptr[chunked_pack] create(
            const table_view & input,
            size_t user_buffer_size,
            cuda_stream_view stream,
            device_memory_resource *temp_mr,
        ) except +libcudf_exception_handler

    cdef struct contiguous_split_result:
        table_view table
        vector[device_buffer] all_data

    cdef vector[contiguous_split_result] contiguous_split (
        table_view input_table,
        vector[size_type] splits
    ) except +libcudf_exception_handler

    cdef packed_columns pack (
        const table_view& input
    ) except +libcudf_exception_handler

    cdef table_view unpack (
        const packed_columns& input
    ) except +libcudf_exception_handler

    cdef table_view unpack (
        const uint8_t* metadata,
        const uint8_t* gpu_data
    ) except +libcudf_exception_handler
