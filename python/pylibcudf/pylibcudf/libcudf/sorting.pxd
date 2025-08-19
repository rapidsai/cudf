# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.aggregation cimport rank_method
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport (
    order,
    null_order,
    null_policy,
    null_order,
    size_type
)
from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "cudf/sorting.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] sorted_order(
        table_view source_table,
        vector[order] column_order,
        vector[null_order] null_precedence,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] stable_sorted_order(
        table_view source_table,
        vector[order] column_order,
        vector[null_order] null_precedence,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] rank(
        column_view input_view,
        rank_method method,
        order column_order,
        null_policy null_handling,
        null_order null_precedence,
        bool percentage,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef bool is_sorted(
        const table_view& table,
        vector[order] column_order,
        vector[null_order] null_precedence,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] segmented_sort_by_key(
        const table_view& values,
        const table_view& keys,
        const column_view& segment_offsets,
        vector[order] column_order,
        vector[null_order] null_precedence,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] stable_segmented_sort_by_key(
        const table_view& values,
        const table_view& keys,
        const column_view& segment_offsets,
        vector[order] column_order,
        vector[null_order] null_precedence,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] sort_by_key(
        const table_view& values,
        const table_view& keys,
        vector[order] column_order,
        vector[null_order] null_precedence,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] stable_sort_by_key(
        const table_view& values,
        const table_view& keys,
        vector[order] column_order,
        vector[null_order] null_precedence,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] sort(
        table_view source_table,
        vector[order] column_order,
        vector[null_order] null_precedence,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] stable_sort(
        table_view source_table,
        vector[order] column_order,
        vector[null_order] null_precedence,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] top_k(
        const column_view& col,
        size_type k,
        order sort_order,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] top_k_order(
        const column_view& col,
        size_type k,
        order sort_order,
        cuda_stream_view stream
    ) except +libcudf_exception_handler
