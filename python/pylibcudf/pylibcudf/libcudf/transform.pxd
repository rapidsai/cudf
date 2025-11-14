# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport bitmask_type, data_type, size_type, null_aware

from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/transform.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[device_buffer], size_type] bools_to_mask (
        const column_view& input,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] mask_to_bools (
        const bitmask_type* bitmask,
        size_type begin_bit,
        size_type end_bit,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[device_buffer], size_type] nans_to_nulls(
        const column_view& input,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] compute_column(
        table_view table,
        expression expr,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] transform(
        const vector[column_view] & inputs,
        const string & transform_udf,
        data_type output_type,
        bool is_ptx,
        optional[void *] user_data,
        null_aware is_null_aware,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[table], unique_ptr[column]] encode(
        table_view input,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[column], table_view] one_hot_encode(
        column_view input_column,
        column_view categories,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] compute_column(
        const table_view table,
        const expression& expr,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
