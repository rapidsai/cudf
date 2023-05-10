# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib.cpp.merge cimport merge as cpp_merge
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.utils cimport columns_from_unique_ptr, table_view_from_columns


def merge_sorted(
    list input_columns,
    list key_columns_indices,
    bool ascending=True,
    str na_position="last",
):
    """Merge multiple lists of lexicographically sorted columns into one list
    of sorted columns. `input_columns` is a list of lists of columns to be
    merged.
    """
    cdef vector[libcudf_types.size_type] c_column_keys = key_columns_indices
    cdef vector[table_view] c_input_tables
    cdef vector[libcudf_types.order] c_column_order
    cdef vector[libcudf_types.null_order] c_null_precedence

    c_input_tables.reserve(len(input_columns))
    for source_columns in input_columns:
        c_input_tables.push_back(
            table_view_from_columns(source_columns))

    num_keys = len(key_columns_indices)

    cdef libcudf_types.order column_order = (
        libcudf_types.order.ASCENDING if ascending
        else libcudf_types.order.DESCENDING
    )
    c_column_order = vector[libcudf_types.order](num_keys, column_order)

    if not ascending:
        na_position = "last" if na_position == "first" else "first"
    cdef libcudf_types.null_order null_precedence = (
        libcudf_types.null_order.BEFORE if na_position == "first"
        else libcudf_types.null_order.AFTER
    )
    c_null_precedence = vector[libcudf_types.null_order](
        num_keys,
        null_precedence
    )

    # Perform sorted merge operation
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_merge(
                c_input_tables,
                c_column_keys,
                c_column_order,
                c_null_precedence,
            )
        )

    return columns_from_unique_ptr(move(c_result))
