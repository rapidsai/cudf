# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.types import cudf_to_np_types, np_to_cudf_types

cimport cudf._lib.pylibcudf.libcudf.types as libcudf_types
from cudf._lib.pylibcudf.libcudf.aggregation cimport rank_method
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view


cdef extern from "cudf/sorting.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] sorted_order(
        table_view source_table,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +

    cdef unique_ptr[column] stable_sorted_order(
        table_view source_table,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +

    cdef unique_ptr[column] rank(
        column_view input_view,
        rank_method method,
        libcudf_types.order column_order,
        libcudf_types.null_policy null_handling,
        libcudf_types.null_order null_precedence,
        bool percentage) except +

    cdef bool is_sorted(
        const table_view& table,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +

    cdef unique_ptr[table] segmented_sort_by_key(
        const table_view& values,
        const table_view& keys,
        const column_view& segment_offsets,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +

    cdef unique_ptr[table] stable_segmented_sort_by_key(
        const table_view& values,
        const table_view& keys,
        const column_view& segment_offsets,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +

    cdef unique_ptr[table] sort_by_key(
        const table_view& values,
        const table_view& keys,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +

    cdef unique_ptr[table] stable_sort_by_key(
        const table_view& values,
        const table_view& keys,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +

    cdef unique_ptr[table] sort(
        table_view source_table,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +

    cdef unique_ptr[table] stable_sort(
        table_view source_table,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +
