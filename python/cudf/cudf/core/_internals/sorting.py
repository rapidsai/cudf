# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal

import pylibcudf as plc

from cudf._lib.column import Column
from cudf.core.buffer import acquire_spill_lock

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cudf.core.column import ColumnBase


@acquire_spill_lock()
def is_sorted(
    source_columns: list[ColumnBase],
    ascending: list[bool] | None = None,
    null_position: list[bool] | None = None,
) -> bool:
    """
    Checks whether the rows of a `table` are sorted in lexicographical order.

    Parameters
    ----------
    source_columns : list of columns
        columns to be checked for sort order
    ascending : None or list-like of booleans
        None or list-like of boolean values indicating expected sort order of
        each column. If list-like, size of list-like must be len(columns). If
        None, all columns expected sort order is set to ascending. False (0) -
        descending, True (1) - ascending.
    null_position : None or list-like of booleans
        None or list-like of boolean values indicating desired order of nulls
        compared to other elements. If list-like, size of list-like must be
        len(columns). If None, null order is set to before. False (0) - after,
        True (1) - before.

    Returns
    -------
    returns : boolean
        Returns True, if sorted as expected by ``ascending`` and
        ``null_position``, False otherwise.
    """
    if ascending is None:
        column_order = [plc.types.Order.ASCENDING] * len(source_columns)
    else:
        if len(ascending) != len(source_columns):
            raise ValueError(
                f"Expected a list-like of length {len(source_columns)}, "
                f"got length {len(ascending)} for `ascending`"
            )
        column_order = [
            plc.types.Order.ASCENDING if asc else plc.types.Order.DESCENDING
            for asc in ascending
        ]

    if null_position is None:
        null_precedence = [plc.types.NullOrder.AFTER] * len(source_columns)
    else:
        if len(null_position) != len(source_columns):
            raise ValueError(
                f"Expected a list-like of length {len(source_columns)}, "
                f"got length {len(null_position)} for `null_position`"
            )
        null_precedence = [
            plc.types.NullOrder.BEFORE if null else plc.types.NullOrder.AFTER
            for null in null_position
        ]

    return plc.sorting.is_sorted(
        plc.Table([col.to_pylibcudf(mode="read") for col in source_columns]),
        column_order,
        null_precedence,
    )


def ordering(
    column_order: list[bool],
    null_precedence: Iterable[Literal["first", "last"]],
) -> tuple[list[plc.types.Order], list[plc.types.NullOrder]]:
    """
    Construct order and null order vectors

    Parameters
    ----------
    column_order
        Iterable of bool (True for ascending order, False for descending)
    null_precedence
        Iterable string for null positions ("first" for start, "last" for end)

    Both iterables must be the same length (not checked)

    Returns
    -------
    pair of vectors (order, and null_order)
    """
    c_column_order = []
    c_null_precedence = []
    for asc, null in zip(column_order, null_precedence):
        c_column_order.append(
            plc.types.Order.ASCENDING if asc else plc.types.Order.DESCENDING
        )
        if asc ^ (null == "first"):
            c_null_precedence.append(plc.types.NullOrder.AFTER)
        elif asc ^ (null == "last"):
            c_null_precedence.append(plc.types.NullOrder.BEFORE)
        else:
            raise ValueError(f"Invalid null precedence {null}")
    return c_column_order, c_null_precedence


@acquire_spill_lock()
def order_by(
    columns_from_table: list[ColumnBase],
    ascending: list[bool],
    na_position: Literal["first", "last"],
    *,
    stable: bool,
):
    """
    Get index to sort the table in ascending/descending order.

    Parameters
    ----------
    columns_from_table : list[Column]
        Columns from the table which will be sorted
    ascending : sequence[bool]
         Sequence of boolean values which correspond to each column
         in the table to be sorted signifying the order of each column
         True - Ascending and False - Descending
    na_position : str
        Whether null values should show up at the "first" or "last"
        position of **all** sorted column.
    stable : bool
        Should the sort be stable? (no default)

    Returns
    -------
    Column of indices that sorts the table
    """
    order = ordering(ascending, itertools.repeat(na_position))
    func = (
        plc.sorting.stable_sorted_order if stable else plc.sorting.sorted_order
    )
    return Column.from_pylibcudf(
        func(
            plc.Table(
                [col.to_pylibcudf(mode="read") for col in columns_from_table],
            ),
            order[0],
            order[1],
        )
    )


@acquire_spill_lock()
def sort_by_key(
    values: list[ColumnBase],
    keys: list[ColumnBase],
    ascending: list[bool],
    na_position: list[Literal["first", "last"]],
    *,
    stable: bool,
) -> list[ColumnBase]:
    """
    Sort a table by given keys

    Parameters
    ----------
    values : list[Column]
        Columns of the table which will be sorted
    keys : list[Column]
        Columns making up the sort key
    ascending : list[bool]
        Sequence of boolean values which correspond to each column
        in the table to be sorted signifying the order of each column
        True - Ascending and False - Descending
    na_position : list[str]
        Sequence of "first" or "last" values (default "first")
        indicating the position of null values when sorting the keys.
    stable : bool
        Should the sort be stable? (no default)

    Returns
    -------
    list[Column]
        list of value columns sorted by keys
    """
    order = ordering(ascending, na_position)
    func = (
        plc.sorting.stable_sort_by_key if stable else plc.sorting.sort_by_key
    )
    return [
        Column.from_pylibcudf(col)
        for col in func(
            plc.Table([col.to_pylibcudf(mode="read") for col in values]),
            plc.Table([col.to_pylibcudf(mode="read") for col in keys]),
            order[0],
            order[1],
        ).columns()
    ]
