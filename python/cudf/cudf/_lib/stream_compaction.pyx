# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import pandas as pd

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.copying cimport gather as cpp_gather, out_of_bounds_policy
from cudf._lib.cpp.sorting cimport (
    stable_sorted_order as cpp_stable_sorted_order,
)
from cudf._lib.cpp.stream_compaction cimport (
    apply_boolean_mask as cpp_apply_boolean_mask,
    drop_duplicates as cpp_drop_duplicates,
    drop_nulls as cpp_drop_nulls,
    duplicate_keep_option,
    unordered_distinct_count as cpp_unordered_distinct_count,
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport (
    nan_policy,
    null_equality,
    null_order,
    null_policy,
    order,
    size_type,
)
from cudf._lib.utils cimport (
    columns_from_unique_ptr,
    data_from_unique_ptr,
    table_view_from_columns,
    table_view_from_table,
)


def drop_nulls(columns: list, how="any", keys=None, thresh=None):
    """
    Drops null rows from cols depending on key columns.

    Parameters
    ----------
    columns : list of columns
    how  : "any" or "all". If thresh is None, drops rows of cols that have any
           nulls or all nulls (respectively) in subset (default: "any")
    keys : List of column indices. If set, then these columns are checked for
           nulls rather than all of columns (optional)
    thresh : Minimum number of non-nulls required to keep a row (optional)

    Returns
    -------
    columns with null rows dropped
    """

    cdef vector[size_type] cpp_keys = (
        keys if keys is not None else range(len(columns))
    )

    cdef size_type c_keep_threshold = cpp_keys.size()
    if thresh is not None:
        c_keep_threshold = thresh
    elif how == "all":
        c_keep_threshold = 1

    cdef unique_ptr[table] c_result
    cdef table_view source_table_view = table_view_from_columns(columns)

    with nogil:
        c_result = move(
            cpp_drop_nulls(
                source_table_view,
                cpp_keys,
                c_keep_threshold
            )
        )

    return columns_from_unique_ptr(move(c_result))


def apply_boolean_mask(columns: list, Column boolean_mask):
    """
    Drops the rows which correspond to False in boolean_mask.

    Parameters
    ----------
    columns : list of columns whose rows are dropped as per boolean_mask
    boolean_mask : a boolean column of same size as source_table

    Returns
    -------
    columns obtained from applying mask
    """

    cdef unique_ptr[table] c_result
    cdef table_view source_table_view = table_view_from_columns(columns)
    cdef column_view boolean_mask_view = boolean_mask.view()

    with nogil:
        c_result = move(
            cpp_apply_boolean_mask(
                source_table_view,
                boolean_mask_view
            )
        )

    return columns_from_unique_ptr(move(c_result))


def drop_duplicates(columns: list,
                    object keys=None,
                    object keep='first',
                    bool nulls_are_equal=True):
    """
    Drops rows in source_table as per duplicate rows in keys.

    Parameters
    ----------
    columns : List of columns
    keys : List of column indices. If set, then these columns are checked for
           duplicates rather than all of columns (optional)
    keep : keep 'first' or 'last' or none of the duplicate rows
    nulls_are_equal : if True, nulls are treated equal else not.

    Returns
    -------
    columns with duplicate dropped
    """

    cdef vector[size_type] cpp_keys = (
        keys if keys is not None else range(len(columns))
    )
    cdef duplicate_keep_option cpp_keep_option

    if keep == 'first':
        cpp_keep_option = duplicate_keep_option.KEEP_FIRST
    elif keep == 'last':
        cpp_keep_option = duplicate_keep_option.KEEP_LAST
    elif keep is False:
        cpp_keep_option = duplicate_keep_option.KEEP_NONE
    else:
        raise ValueError('keep must be either "first", "last" or False')

    # shifting the index number by number of index columns
    cdef null_equality cpp_nulls_equal = (
        null_equality.EQUAL
        if nulls_are_equal
        else null_equality.UNEQUAL
    )

    cdef vector[order] column_order
    column_order.reserve(cpp_keys.size())
    cdef vector[null_order] null_precedence
    null_precedence.reserve(cpp_keys.size())

    for _ in range(cpp_keys.size()):
        column_order.push_back(order.ASCENDING)
        null_precedence.push_back(null_order.BEFORE)

    cdef unique_ptr[column] gather_map
    cdef unique_ptr[table] sorted_source_table
    cdef unique_ptr[table] c_result
    cdef table_view source_table_view = table_view_from_columns(columns)
    cdef table_view keys_view = source_table_view.select(cpp_keys)
    cdef out_of_bounds_policy policy = out_of_bounds_policy.DONT_CHECK

    with nogil:
        gather_map = move(
            cpp_stable_sorted_order(
                keys_view,
                column_order,
                null_precedence
            )
        )
        sorted_source_table = move(
            cpp_gather(
                source_table_view,
                gather_map.get().view(),
                policy
            )
        )
        c_result = move(
            cpp_drop_duplicates(
                sorted_source_table.get().view(),
                cpp_keys,
                cpp_keep_option,
                cpp_nulls_equal
            )
        )

    return columns_from_unique_ptr(move(c_result))


def distinct_count(Column source_column, ignore_nulls=True, nan_as_null=False):
    """
    Finds number of unique rows in `source_column`

    Parameters
    ----------
    source_column : source table checked for unique rows
    ignore_nulls : If True nulls are ignored,
                   else counted as one more distinct value
    nan_as_null  : If True, NAN is considered NULL,
                   else counted as one more distinct value

    Returns
    -------
    Count of number of unique rows in `source_column`
    """

    cdef null_policy cpp_null_handling = (
        null_policy.EXCLUDE
        if ignore_nulls
        else null_policy.INCLUDE
    )
    cdef nan_policy cpp_nan_handling = (
        nan_policy.NAN_IS_NULL
        if nan_as_null
        else nan_policy.NAN_IS_VALID
    )

    cdef column_view source_column_view = source_column.view()
    with nogil:
        count = cpp_unordered_distinct_count(
            source_column_view,
            cpp_null_handling,
            cpp_nan_handling
        )

    return count
