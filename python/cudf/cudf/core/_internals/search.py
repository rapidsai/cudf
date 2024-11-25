# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pylibcudf as plc

from cudf._lib.column import Column
from cudf.core.buffer import acquire_spill_lock

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase


@acquire_spill_lock()
def search_sorted(
    source: list[ColumnBase],
    values: list[ColumnBase],
    side: Literal["left", "right"],
    ascending: bool = True,
    na_position: Literal["first", "last"] = "last",
) -> ColumnBase:
    """Find indices where elements should be inserted to maintain order

    Parameters
    ----------
    source : list of columns
        List of columns to search in
    values : List of columns
        List of value columns to search for
    side : str {'left', 'right'} optional
        If 'left', the index of the first suitable location is given.
        If 'right', return the last such index
    """
    # Note: We are ignoring index columns here
    column_order = [
        plc.types.Order.ASCENDING if ascending else plc.types.Order.DESCENDING
    ] * len(source)
    null_precedence = [
        plc.types.NullOrder.AFTER
        if na_position == "last"
        else plc.types.NullOrder.BEFORE
    ] * len(source)

    func = getattr(
        plc.search,
        "lower_bound" if side == "left" else "upper_bound",
    )
    return Column.from_pylibcudf(
        func(
            plc.Table([col.to_pylibcudf(mode="read") for col in source]),
            plc.Table([col.to_pylibcudf(mode="read") for col in values]),
            column_order,
            null_precedence,
        )
    )
