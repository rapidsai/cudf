# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A column, with some properties."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import cudf._lib.pylibcudf as plc

if TYPE_CHECKING:
    from typing_extensions import Self

__all__: list[str] = ["Column"]


class Column:
    """A column, a name, and sortedness."""

    obj: plc.Column
    name: str
    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder

    def __init__(self, column: plc.Column, name: str):
        self.obj = column
        self.name = name
        self.is_sorted = plc.types.Sorted.NO
        self.order = plc.types.Order.ASCENDING
        self.null_order = plc.types.NullOrder.BEFORE

    def rename(self, name: str) -> Column:
        """Return a new column sharing data with a new name."""
        return type(self)(self.obj, name).with_sorted(like=self)

    def with_sorted(self, *, like: Column) -> Self:
        """Copy sortedness properties from a column onto self."""
        return self.set_sorted(
            is_sorted=like.is_sorted, order=like.order, null_order=like.null_order
        )

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

    def copy(self) -> Self:
        """Return a shallow copy of the column."""
        return type(self)(self.obj, self.name).with_sorted(like=self)

    def mask_nans(self) -> Self:
        """Return a copy of self with nans masked out."""
        if self.nan_count > 0:
            raise NotImplementedError
        else:
            return self.copy()

    @functools.cached_property
    def nan_count(self) -> int:
        """Return the number of NaN values in the column."""
        if self.obj.type().id() not in (plc.TypeId.FLOAT32, plc.TypeId.FLOAT64):
            return 0
        else:
            return plc.reduce.reduce(
                plc.unary.is_nan(self.obj),
                plc.aggregation.sum(),
                # TODO: pylibcudf needs to have a SizeType DataType singleton
                plc.DataType(plc.TypeId.INT32),
            )
