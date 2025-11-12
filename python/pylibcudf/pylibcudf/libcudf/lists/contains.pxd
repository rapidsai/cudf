# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/lists/contains.hpp" namespace "cudf::lists" nogil:

    cpdef enum class duplicate_find_option(int32_t):
        FIND_FIRST
        FIND_LAST

    cdef unique_ptr[column] contains(
        const lists_column_view& lists,
        const scalar& search_key,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] contains(
        const lists_column_view& lists,
        const column_view& search_keys,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] contains_nulls(
        const lists_column_view& lists,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] index_of(
        const lists_column_view& lists,
        const scalar& search_key,
        duplicate_find_option find_option,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] index_of(
        const lists_column_view& lists,
        const column_view& search_keys,
        duplicate_find_option find_option,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
