# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp cimport bool
from pylibcudf.libcudf.aggregation cimport rank_method
from pylibcudf.libcudf.types cimport null_order, null_policy, order, size_type
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .table cimport Table


cpdef Column sorted_order(
    Table source_table, list column_order, list null_precedence, Stream stream=*
)

cpdef Column stable_sorted_order(
    Table source_table,
    list column_order,
    list null_precedence,
    Stream stream=*
)

cpdef Column rank(
    Column input_view,
    rank_method method,
    order column_order,
    null_policy null_handling,
    null_order null_precedence,
    bool percentage,
    Stream stream=*
)

cpdef bool is_sorted(
    Table table, list column_order, list null_precedence, Stream stream=*
)

cpdef Table segmented_sort_by_key(
    Table values,
    Table keys,
    Column segment_offsets,
    list column_order,
    list null_precedence,
    Stream stream=*
)

cpdef Table stable_segmented_sort_by_key(
    Table values,
    Table keys,
    Column segment_offsets,
    list column_order,
    list null_precedence,
    Stream stream=*
)

cpdef Table sort_by_key(
    Table values,
    Table keys,
    list column_order,
    list null_precedence,
    Stream stream=*
)

cpdef Table stable_sort_by_key(
    Table values,
    Table keys,
    list column_order,
    list null_precedence,
    Stream stream=*
)

cpdef Table sort(
    Table source_table, list column_order, list null_precedence, Stream stream=*
)

cpdef Table stable_sort(
    Table source_table, list column_order, list null_precedence, Stream stream=*
)

cpdef Column top_k(Column col, size_type k, order sort_order=*, Stream stream=*)

cpdef Column top_k_order(Column col, size_type k, order sort_order=*, Stream stream=*)
