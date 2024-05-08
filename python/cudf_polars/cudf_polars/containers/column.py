# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A column, with some properties."""

from __future__ import annotations

import cudf._lib.pylibcudf as plc

__all__: list[str] = ["Column"]


class Column:
    """A column, a name, and sortedness."""

    __slots__ = ("obj", "name", "is_sorted", "order", "null_order")
    obj: plc.Column
    name: str
    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder

    def __init__(self, column: plc.Column, name: str):
        self.obj = column
        self.name = name
        self.is_sorted = plc.types.Sorted.NO

    def set_sorted(
        self,
        is_sorted: plc.types.Sorted,
        order: plc.types.Order,
        null_order: plc.types.NullOrder,
    ) -> Column:
        """
        Return a new column sharing data with sortedness set.

        Parameters
        ----------
        is_sorted
            Is the column sorted
        order
            The order if sorted
        null_order
            Where nulls sort, if sorted

        Returns
        -------
        New column sharing data.
        """
        obj = Column(self.obj, self.name)
        obj.is_sorted = is_sorted
        obj.order = order
        obj.null_order = null_order
        return obj
