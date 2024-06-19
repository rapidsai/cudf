# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf._lib.pylibcudf.libcudf.aggregation cimport rank_method
from cudf._lib.pylibcudf.libcudf.types cimport (
    null_order,
    null_policy,
    order,
    size_type,
)

from .column cimport Column
from .table cimport Table


cpdef Column sorted_order(Table source_table, list column_order, list null_precedence)

cpdef Column stable_sorted_order(
    Table source_table,
    list column_order,
    list null_precedence,
)

cpdef Column rank(
    Column input_view,
    rank_method method,
    order column_order,
    null_policy null_handling,
    null_order null_precedence,
    bool percentage,
)

cpdef bool is_sorted(Table table, list column_order, list null_precedence)

cpdef Table segmented_sort_by_key(
    Table values,
    Table keys,
    Column segment_offsets,
    list column_order,
    list null_precedence,
)

cpdef Table stable_segmented_sort_by_key(
    Table values,
    Table keys,
    Column segment_offsets,
    list column_order,
    list null_precedence,
)

cpdef Table sort_by_key(
    Table values,
    Table keys,
    list column_order,
    list null_precedence,
)

cpdef Table stable_sort_by_key(
    Table values,
    Table keys,
    list column_order,
    list null_precedence,
)

cpdef Table sort(Table source_table, list column_order, list null_precedence)

cpdef Table stable_sort(Table source_table, list column_order, list null_precedence)
