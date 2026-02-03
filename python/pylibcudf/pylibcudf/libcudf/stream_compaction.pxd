# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport (
    nan_equality,
    nan_policy,
    null_equality,
    null_policy,
    size_type,
)
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/stream_compaction.hpp" namespace "cudf" nogil:
    cpdef enum class duplicate_keep_option:
        KEEP_ANY
        KEEP_FIRST
        KEEP_LAST
        KEEP_NONE

    cdef unique_ptr[table] drop_nulls(
        table_view source_table,
        vector[size_type] keys,
        size_type keep_threshold,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] drop_nans(
        table_view source_table,
        vector[size_type] keys,
        size_type keep_threshold,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] apply_boolean_mask(
        table_view source_table,
        column_view boolean_mask,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] unique(
        table_view input,
        vector[size_type] keys,
        duplicate_keep_option keep,
        null_equality nulls_equal,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] distinct(
        table_view input,
        vector[size_type] keys,
        duplicate_keep_option keep,
        null_equality nulls_equal,
        nan_equality nans_equals,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] distinct_indices(
        table_view input,
        duplicate_keep_option keep,
        null_equality nulls_equal,
        nan_equality nans_equal,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] stable_distinct(
        table_view input,
        vector[size_type] keys,
        duplicate_keep_option keep,
        null_equality nulls_equal,
        nan_equality nans_equal,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef size_type unique_count(
        column_view column,
        null_policy null_handling,
        nan_policy nan_handling,
        cuda_stream_view stream) except +libcudf_exception_handler

    cdef size_type unique_count(
        table_view source_table,
        null_policy null_handling) except +libcudf_exception_handler

    cdef size_type distinct_count(
        column_view column,
        null_policy null_handling,
        nan_policy nan_handling,
        cuda_stream_view stream) except +libcudf_exception_handler

    cdef size_type distinct_count(
        table_view source_table,
        null_policy null_handling) except +libcudf_exception_handler

    cdef unique_ptr[table] filter(
        table_view predicate_table,
        const expression& predicate_expr,
        table_view filter_table,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
