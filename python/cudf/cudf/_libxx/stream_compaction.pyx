# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd
from cudf._libxx.column cimport *
from cudf._libxx.table cimport *

from cudf._libxx.stream_compaction import *
from cudf._libxx.stream_compaction cimport *

from cudf._libxx.stream_compaction cimport (
    duplicate_keep_option,
    drop_nulls as cpp_drop_nulls,
    apply_boolean_mask as cpp_apply_boolean_mask,
    drop_duplicates as cpp_drop_duplicates,
    unique_count as cpp_unique_count
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

    cdef unique_ptr[table] c_result = (
        cpp_drop_nulls(source_table.view(),
                       cpp_keys,
                       c_keep_threshold)
    )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=(
            None if source_table._index is None
            else source_table._index._column_names)
    )


def apply_boolean_mask(Table source_table, Column boolean_mask):
    """
    Drops the rows which correspond to False in boolean_mask.

    Parameters
    ----------
    source_table : source table whose rows are droppped as per boolean_mask
    boolean_mask : a boolean column of same size as source_table

    Returns
    -------
    Table obtained from applying mask
    """

    assert pd.api.types.is_bool_dtype(boolean_mask.dtype)

    cdef unique_ptr[table] c_result = (
        cpp_apply_boolean_mask(source_table.view(),
                               boolean_mask.view())
    )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=(
            None if source_table._index
            is None else source_table._index._column_names)
    )


def drop_duplicates(Table source_table, keys=None,
                    keep='first', nulls_are_equal=True):
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
        0 if source_table._index is None
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

    cdef cpp_nulls_are_equal = nulls_are_equal

    cdef unique_ptr[table] c_result = (
        cpp_drop_duplicates(source_table.view(),
                            cpp_keys,
                            cpp_keep_option,
                            cpp_nulls_are_equal)
    )

    return Table.from_unique_ptr(
        move(c_result),
        column_names=source_table._column_names,
        index_names=(
            None if source_table._index
            is None else source_table._index._column_names)
    )


def unique_count(Column source_column, ignore_nulls=True, nan_as_null=False):
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

    cdef cpp_ignore_nulls = ignore_nulls
    cdef cpp_nan_as_null = nan_as_null

    count = cpp_unique_count(source_column.view(),
                             cpp_ignore_nulls,
                             cpp_nan_as_null)

    return count
