# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool

from cudf._lib.column cimport Column
from cudf._lib.utils cimport columns_from_pylibcudf_table

import pylibcudf


@acquire_spill_lock()
def drop_nulls(list columns, how="any", keys=None, thresh=None):
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
    if how not in {"any", "all"}:
        raise ValueError("how must be 'any' or 'all'")

    keys = list(keys if keys is not None else range(len(columns)))

    # Note: If how == "all" and thresh is specified this prioritizes thresh
    if thresh is not None:
        keep_threshold = thresh
    elif how == "all":
        keep_threshold = 1
    else:
        keep_threshold = len(keys)

    return columns_from_pylibcudf_table(
        pylibcudf.stream_compaction.drop_nulls(
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in columns]),
            keys,
            keep_threshold,
        )
    )


@acquire_spill_lock()
def apply_boolean_mask(list columns, Column boolean_mask):
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
    return columns_from_pylibcudf_table(
        pylibcudf.stream_compaction.apply_boolean_mask(
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in columns]),
            boolean_mask.to_pylibcudf(mode="read"),
        )
    )


_keep_options = {
    "first": pylibcudf.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
    "last": pylibcudf.stream_compaction.DuplicateKeepOption.KEEP_LAST,
    False: pylibcudf.stream_compaction.DuplicateKeepOption.KEEP_NONE,
}


@acquire_spill_lock()
def drop_duplicates(list columns,
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
    if (keep_option := _keep_options.get(keep)) is None:
        raise ValueError('keep must be either "first", "last" or False')

    return columns_from_pylibcudf_table(
        pylibcudf.stream_compaction.stable_distinct(
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in columns]),
            list(keys if keys is not None else range(len(columns))),
            keep_option,
            pylibcudf.types.NullEquality.EQUAL
            if nulls_are_equal else pylibcudf.types.NullEquality.UNEQUAL,
            pylibcudf.types.NanEquality.ALL_EQUAL,
        )
    )


@acquire_spill_lock()
def distinct_indices(
    list columns,
    object keep="first",
    bool nulls_equal=True,
    bool nans_equal=True,
):
    """
    Return indices of the distinct rows in a table.

    Parameters
    ----------
    columns : list of columns to check for duplicates
    keep : treat "first", "last", or (False) none of any duplicate
        rows as distinct
    nulls_equal : Should nulls compare equal
    nans_equal: Should nans compare equal

    Returns
    -------
    Column of indices

    See Also
    --------
    drop_duplicates
    """
    if (keep_option := _keep_options.get(keep)) is None:
        raise ValueError('keep must be either "first", "last" or False')

    return Column.from_pylibcudf(
        pylibcudf.stream_compaction.distinct_indices(
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in columns]),
            keep_option,
            pylibcudf.types.NullEquality.EQUAL
            if nulls_equal else pylibcudf.types.NullEquality.UNEQUAL,
            pylibcudf.types.NanEquality.ALL_EQUAL
            if nans_equal else pylibcudf.types.NanEquality.UNEQUAL,
        )
    )


@acquire_spill_lock()
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
    return pylibcudf.stream_compaction.distinct_count(
        source_column.to_pylibcudf(mode="read"),
        pylibcudf.types.NullPolicy.EXCLUDE
        if ignore_nulls else pylibcudf.types.NullPolicy.INCLUDE,
        pylibcudf.types.NanPolicy.NAN_IS_NULL
        if nan_as_null else pylibcudf.types.NanPolicy.NAN_IS_VALID,
    )
