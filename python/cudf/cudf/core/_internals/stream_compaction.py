# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pylibcudf as plc

from cudf._lib.column import Column
from cudf.core.buffer import acquire_spill_lock

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase


@acquire_spill_lock()
def drop_nulls(
    columns: list[ColumnBase],
    how: Literal["any", "all"] = "any",
    keys: list[int] | None = None,
    thresh: int | None = None,
) -> list[ColumnBase]:
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

    keys = keys if keys is not None else list(range(len(columns)))

    # Note: If how == "all" and thresh is specified this prioritizes thresh
    if thresh is not None:
        keep_threshold = thresh
    elif how == "all":
        keep_threshold = 1
    else:
        keep_threshold = len(keys)

    plc_table = plc.stream_compaction.drop_nulls(
        plc.Table([col.to_pylibcudf(mode="read") for col in columns]),
        keys,
        keep_threshold,
    )
    return [Column.from_pylibcudf(col) for col in plc_table.columns()]


@acquire_spill_lock()
def apply_boolean_mask(
    columns: list[ColumnBase], boolean_mask: ColumnBase
) -> list[ColumnBase]:
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
    plc_table = plc.stream_compaction.apply_boolean_mask(
        plc.Table([col.to_pylibcudf(mode="read") for col in columns]),
        boolean_mask.to_pylibcudf(mode="read"),
    )
    return [Column.from_pylibcudf(col) for col in plc_table.columns()]


@acquire_spill_lock()
def drop_duplicates(
    columns: list[ColumnBase],
    keys: list[int] | None = None,
    keep: Literal["first", "last", False] = "first",
    nulls_are_equal: bool = True,
) -> list[ColumnBase]:
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
    _keep_options = {
        "first": plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        "last": plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
        False: plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
    }
    if (keep_option := _keep_options.get(keep)) is None:
        raise ValueError('keep must be either "first", "last" or False')

    plc_table = plc.stream_compaction.stable_distinct(
        plc.Table([col.to_pylibcudf(mode="read") for col in columns]),
        keys if keys is not None else list(range(len(columns))),
        keep_option,
        plc.types.NullEquality.EQUAL
        if nulls_are_equal
        else plc.types.NullEquality.UNEQUAL,
        plc.types.NanEquality.ALL_EQUAL,
    )
    return [Column.from_pylibcudf(col) for col in plc_table.columns()]
