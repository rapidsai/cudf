# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A column, with some properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cudf._lib.pylibcudf as plc

if TYPE_CHECKING:
    from typing_extensions import Self

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

    def with_metadata(self, *, like: Column) -> Self:
        """Copy metadata from a column onto self."""
        self.is_sorted = like.is_sorted
        self.order = like.order
        self.null_order = like.null_order
        return self

    def set_sorted(
        self,
        *,
        is_sorted: plc.types.Sorted,
        order: plc.types.Order,
        null_order: plc.types.NullOrder,
    ) -> Self:
        """
        Modify sortedness metadata in place.

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
        Self with metadata set.
        """
        self.is_sorted = is_sorted
        self.order = order
        self.null_order = null_order
        return self
