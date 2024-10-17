# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type

from rmm.librmm.device_buffer cimport device_buffer


cdef extern from "cudf/contiguous_split.hpp" namespace "cudf" nogil:
    cdef cppclass packed_columns:
        unique_ptr[vector[uint8_t]] metadata
        unique_ptr[device_buffer] gpu_data

    cdef cppclass packed_table:
        table_view table
        packed_columns data

    cdef cppclass chunked_pack:
        size_type get_total_contiguous_size() except +
        bool has_next() except +
        unique_ptr[vector[uint8_t]] build_metadata() except +

        @staticmethod
        unique_ptr[vector[chunked_pack]] create(
            table_view table,
            size_type buffer_size,
        ) except +

    cdef vector[packed_table] contiguous_split (
        table_view input_table,
        vector[size_type] splits
    ) except +

    cdef packed_columns pack (const table_view& input) except +

    cdef table_view unpack (const packed_columns& input) except +

    cdef table_view unpack (
        const uint8_t* metadata,
        const uint8_t* gpu_data
    ) except +
