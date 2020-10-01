# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.utility cimport move
from libcpp cimport bool

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table

from cudf._lib.cpp.types cimport (
    size_type, null_policy, nan_policy, null_equality
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.stream_compaction cimport (
    duplicate_keep_option,
    drop_nulls as cpp_drop_nulls,
    apply_boolean_mask as cpp_apply_boolean_mask,
    drop_duplicates as cpp_drop_duplicates,
    distinct_count as cpp_distinct_count
)


def drop_nulls(Table source_table, how="any", keys=None, thresh=None):
    """
    Drops null rows from cols depending on key columns.

    Parameters
    ----------
    source_table : source table whose null rows are dropped to form new table
    how  : "any" or "all". If thresh is None, drops rows of cols that have any
           nulls or all nulls (respectively) in subset (default: "any")
    keys : List of Column names. If set, then these columns are checked for
           nulls rather than all of cols (optional)
    thresh : Minimum number of non-nulls required to keep a row (optional)

    Returns
    -------
    Table with null rows dropped
    """

    num_index_columns = (
        0 if source_table._index is None else
        source_table._index._num_columns)
    # shifting the index number by number of index columns
    cdef vector[size_type] cpp_keys = (
        [
            num_index_columns + source_table._column_names.index(name)
            for name in keys
        ]
        if keys is not None
        else range(
            num_index_columns, num_index_columns + source_table._num_columns
        )
    )

    cdef size_type c_keep_threshold = cpp_keys.size()
    if thresh is not None:
        c_keep_threshold = thresh
    elif how == "all":
        c_keep_threshold = 1

    cdef unique_ptr[table] c_result
    cdef table_view source_table_view = source_table.view()

    with nogil:
        c_result = move(
            cpp_drop_nulls(
                source_table_view,
                cpp_keys,
                c_keep_threshold
            )
        )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=(
            None if source_table._index is None
            else source_table._index_names)
    )


def apply_boolean_mask(Table source_table, Column boolean_mask):
    """
    Drops the rows which correspond to False in boolean_mask.

    Parameters
    ----------
    source_table : source table whose rows are dropped as per boolean_mask
    boolean_mask : a boolean column of same size as source_table

    Returns
    -------
    Table obtained from applying mask
    """

    assert pd.api.types.is_bool_dtype(boolean_mask.dtype)

    cdef unique_ptr[table] c_result
    cdef table_view source_table_view = source_table.view()
    cdef column_view boolean_mask_view = boolean_mask.view()

    with nogil:
        c_result = move(
            cpp_apply_boolean_mask(
                source_table_view,
                boolean_mask_view
            )
        )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=(
            None if source_table._index
            is None else source_table._index_names)
    )


def drop_duplicates(Table source_table,
                    object keys=None,
                    object keep='first',
                    bool nulls_are_equal=True,
                    bool ignore_index=False):
    """
    Drops rows in source_table as per duplicate rows in keys.

    Parameters
    ----------
    source_table : source_table whose rows gets dropped
    keys : List of Column names belong to source_table
    keep : keep 'first' or 'last' or none of the duplicate rows
    nulls_are_equal : if True, nulls are treated equal else not.

    Returns
    -------
    Table with duplicate dropped
    """

    cdef duplicate_keep_option cpp_keep_option

    if keep == 'first':
        cpp_keep_option = duplicate_keep_option.KEEP_FIRST
    elif keep == 'last':
        cpp_keep_option = duplicate_keep_option.KEEP_LAST
    elif keep is False:
        cpp_keep_option = duplicate_keep_option.KEEP_NONE
    else:
        raise ValueError('keep must be either "first", "last" or False')

    num_index_columns =(
        0 if (source_table._index is None or ignore_index)
        else source_table._index._num_columns)
    # shifting the index number by number of index columns
    cdef vector[size_type] cpp_keys = (
        [
            num_index_columns + source_table._column_names.index(name)
            for name in keys
        ]
        if keys is not None
        else range(
            num_index_columns, num_index_columns + source_table._num_columns
        )
    )

    cdef null_equality cpp_nulls_equal = (
        null_equality.EQUAL
        if nulls_are_equal
        else null_equality.UNEQUAL
    )
    cdef unique_ptr[table] c_result
    cdef table_view source_table_view
    if ignore_index:
        source_table_view = source_table.data_view()
    else:
        source_table_view = source_table.view()

    with nogil:
        c_result = move(
            cpp_drop_duplicates(
                source_table_view,
                cpp_keys,
                cpp_keep_option,
                cpp_nulls_equal
            )
        )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=(
            None if (source_table._index is None or ignore_index)
            else source_table._index_names)
    )


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
        count = cpp_distinct_count(
            source_column_view,
            cpp_null_handling,
            cpp_nan_handling
        )

    return count
