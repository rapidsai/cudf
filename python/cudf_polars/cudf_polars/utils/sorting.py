# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Sorting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cudf._lib.pylibcudf as plc

if TYPE_CHECKING:
    from collections.abc import Sequence


def sort_order(
    descending: Sequence[bool], *, nulls_last: bool, num_keys: int
) -> tuple[list[plc.types.Order], list[plc.types.NullOrder]]:
    """
    Produce sort order arguments.

    Parameters
    ----------
    descending
        List indicating order for each column
    nulls_last
        Should nulls sort last or first?
    num_keys
        Number of sort keys

    Returns
    -------
    tuple of column_order and null_precendence
    suitable for passing to sort routines
    """
    # Mimicking polars broadcast handling of descending
    if num_keys > (n := len(descending)) and n == 1:
        descending = [descending[0]] * num_keys
    column_order = [
        plc.types.Order.DESCENDING if d else plc.types.Order.ASCENDING
        for d in descending
    ]
    null_precedence = []
    for asc in column_order:
        if (asc == plc.types.Order.ASCENDING) ^ (not nulls_last):
            null_precedence.append(plc.types.NullOrder.AFTER)
        elif (asc == plc.types.Order.ASCENDING) ^ nulls_last:
            null_precedence.append(plc.types.NullOrder.BEFORE)
    return column_order, null_precedence
