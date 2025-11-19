# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pylibcudf as plc

from cudf.core.buffer import acquire_spill_lock

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cudf.core.column import ColumnBase


@acquire_spill_lock()
def is_sorted(
    source_columns: Iterable[ColumnBase],
    ascending: Iterable[bool],
    na_position: Iterable[Literal["first", "last"]],
) -> bool:
    """
    Checks whether the rows of a `table` are sorted in lexicographical order.

    Parameters
    ----------
    source_columns : iterable of columns
        columns to be checked for sort order
    ascending : list-like of booleans
        list-like of boolean values indicating expected sort order of
        each column. If list-like, size of list-like must be len(columns). If
        None, all columns expected sort order is set to ascending. False (0) -
        descending, True (1) - ascending.
    na_position : list-like of booleans
        list-like of boolean values indicating desired order of nulls
        compared to other elements. If list-like, size of list-like must be
        len(columns). If None, null order is set to before. False (0) - after,
        True (1) - before.

    Returns
    -------
    returns : boolean
        Returns True, if sorted as expected by ``ascending`` and
        ``null_position``, False otherwise.
    """
    column_order, null_precedence = ordering(ascending, na_position)
    return plc.sorting.is_sorted(
        plc.Table([col.to_pylibcudf(mode="read") for col in source_columns]),
        column_order,
        null_precedence,
    )


def ordering(
    ascending: Iterable[bool],
    na_position: Iterable[Literal["first", "last"]],
) -> tuple[list[plc.types.Order], list[plc.types.NullOrder]]:
    """
    Convert bool ascending and string na_position to
    plc.types.Order and plc.types.NullOrder, respectively.

    Parameters
    ----------
    ascending
        Iterable of bool (True for ascending order, False for descending)
    na_position
        Iterable string for null positions ("first" for start, "last" for end)

    Both iterables must be the same length

    Returns
    -------
    pair of vectors (order, and null_order)
    """
    c_column_order = []
    c_null_precedence = []
    for asc, null in zip(ascending, na_position, strict=True):
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
    columns_from_table: Iterable[ColumnBase],
    ascending: Iterable[bool],
    na_position: Iterable[Literal["first", "last"]],
    *,
    stable: bool,
) -> plc.Column:
    """
    Get index to sort the table in ascending/descending order.

    Parameters
    ----------
    columns_from_table : Iterable[Column]
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
    column_order, null_precedence = ordering(ascending, na_position)
    func = (
        plc.sorting.stable_sorted_order if stable else plc.sorting.sorted_order
    )
    return func(
        plc.Table(
            [col.to_pylibcudf(mode="read") for col in columns_from_table],
        ),
        column_order,
        null_precedence,
    )


@acquire_spill_lock()
def sort_by_key(
    values: Iterable[ColumnBase],
    keys: Iterable[ColumnBase],
    ascending: Iterable[bool],
    na_position: Iterable[Literal["first", "last"]],
    *,
    stable: bool,
) -> list[plc.Column]:
    """
    Sort a table by given keys

    Parameters
    ----------
    values : Iterable[Column]
        Columns of the table which will be sorted
    keys : Iterable[Column]
        Columns making up the sort key
    ascending : Iterable[bool]
        Sequence of boolean values which correspond to each column
        in the table to be sorted signifying the order of each column
        True - Ascending and False - Descending
    na_position : Iterable[str]
        Sequence of "first" or "last" values (default "first")
        indicating the position of null values when sorting the keys.
    stable : bool
        Should the sort be stable? (no default)

    Returns
    -------
    list[Column]
        list of value columns sorted by keys
    """
    column_order, null_precedence = ordering(ascending, na_position)
    func = (
        plc.sorting.stable_sort_by_key if stable else plc.sorting.sort_by_key
    )
    return func(
        plc.Table([col.to_pylibcudf(mode="read") for col in values]),
        plc.Table([col.to_pylibcudf(mode="read") for col in keys]),
        column_order,
        null_precedence,
    ).columns()


@acquire_spill_lock()
def search_sorted(
    source: Iterable[ColumnBase],
    values: Iterable[ColumnBase],
    side: Literal["left", "right"],
    ascending: Iterable[bool],
    na_position: Iterable[Literal["first", "last"]],
) -> plc.Column:
    """Find indices where elements should be inserted to maintain order

    Parameters
    ----------
    source : Iterable of columns
        Iterable of columns to search in
    values : Iterable of columns
        Iterable of value columns to search for
    side : str {'left', 'right'} optional
        If 'left', the index of the first suitable location is given.
        If 'right', return the last such index
    ascending : Iterable[bool]
        Iterable of bools which correspond to each column's
        sort order.
    na_position : Iterable[str]
        Iterable of strings which correspond to each column's
        null position.
    """
    column_order, null_precedence = ordering(ascending, na_position)
    func = getattr(
        plc.search,
        "lower_bound" if side == "left" else "upper_bound",
    )
    return func(
        plc.Table([col.to_pylibcudf(mode="read") for col in source]),
        plc.Table([col.to_pylibcudf(mode="read") for col in values]),
        column_order,
        null_precedence,
    )
