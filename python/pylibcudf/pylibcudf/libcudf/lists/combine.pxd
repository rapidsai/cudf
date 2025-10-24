# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/lists/combine.hpp" namespace \
        "cudf::lists" nogil:

    cpdef enum class concatenate_null_policy(int32_t):
        IGNORE
        NULLIFY_OUTPUT_ROW

    cdef unique_ptr[column] concatenate_rows(
        const table_view input_table,
        concatenate_null_policy null_policy,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] concatenate_list_elements(
        const table_view input_table,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] concatenate_list_elements(
        const column_view input_table,
        concatenate_null_policy null_policy,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
