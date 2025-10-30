# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from pylibcudf.libcudf.types cimport data_type, size_type

from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/column/column.hpp" namespace "cudf" nogil:
    cdef cppclass column_contents "cudf::column::contents":
        unique_ptr[device_buffer] data
        unique_ptr[device_buffer] null_mask
        vector[unique_ptr[column]] children

    cdef cppclass column:
        column() except +libcudf_exception_handler
        column(
            const column& other,
            cuda_stream_view stream,
            device_memory_resource* mr
        ) except +libcudf_exception_handler

        column(
            column_view view,
            cuda_stream_view stream,
            device_memory_resource* mr
        ) except +libcudf_exception_handler

        size_type size() except +libcudf_exception_handler
        size_type null_count() except +libcudf_exception_handler
        bool has_nulls() except +libcudf_exception_handler
        data_type type() except +libcudf_exception_handler
        column_view view() except +libcudf_exception_handler
        mutable_column_view mutable_view() except +libcudf_exception_handler
        column_contents release() except +libcudf_exception_handler
