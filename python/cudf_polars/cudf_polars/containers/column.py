# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A column, with some properties."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import cudf._lib.pylibcudf as plc

if TYPE_CHECKING:
    from typing_extensions import Self

__all__: list[str] = ["Column", "NamedColumn"]


class Column:
    """An immutable column with sortedness metadata."""

    obj: plc.Column
    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder
    is_scalar: bool

    def __init__(
        self,
        column: plc.Column,
        *,
        is_sorted: plc.types.Sorted = plc.types.Sorted.NO,
        order: plc.types.Order = plc.types.Order.ASCENDING,
        null_order: plc.types.NullOrder = plc.types.NullOrder.BEFORE,
    ):
        self.obj = column
        self.is_scalar = self.obj.size() == 1
        if self.obj.size() <= 1:
            is_sorted = plc.types.Sorted.YES
        self.is_sorted = is_sorted
        self.order = order
        self.null_order = null_order

    @functools.cached_property
    def obj_scalar(self) -> plc.Scalar:
        """
        A copy of the column object as a pylibcudf Scalar.

        Returns
        -------
        pylibcudf Scalar object.

        Raises
        ------
        ValueError
            If the column is not length-1.
        """
        if not self.is_scalar:
            raise ValueError(
                f"Cannot convert a column of length {self.obj.size()} to scalar"
            )
        return plc.copying.get_element(self.obj, 0)

    def sorted_like(self, like: Column, /) -> Self:
        """
        Copy sortedness properties from a column onto self.

        Parameters
        ----------
        like
            The column to copy sortedness metadata from.

        Returns
        -------
        Self with metadata set.

        See Also
        --------
        set_sorted
        """
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
        if self.obj.size() <= 1:
            is_sorted = plc.types.Sorted.YES
        self.is_sorted = is_sorted
        self.order = order
        self.null_order = null_order
        return self

    def copy(self) -> Self:
        """
        A shallow copy of the column.

        Returns
        -------
        New column sharing data with self.
        """
        return type(self)(
            self.obj,
            is_sorted=self.is_sorted,
            order=self.order,
            null_order=self.null_order,
        )

    def mask_nans(self) -> Self:
        """Return a shallow copy of self with nans masked out."""
        if plc.traits.is_floating_point(self.obj.type()):
            old_count = self.obj.null_count()
            mask, new_count = plc.transform.nans_to_nulls(self.obj)
            result = type(self)(self.obj.with_mask(mask, new_count))
            if old_count == new_count:
                return result.sorted_like(self)
            return result
        return self.copy()

    @functools.cached_property
    def nan_count(self) -> int:
        """Return the number of NaN values in the column."""
        if plc.traits.is_floating_point(self.obj.type()):
            return plc.interop.to_arrow(
                plc.reduce.reduce(
                    plc.unary.is_nan(self.obj),
                    plc.aggregation.sum(),
                    # TODO: pylibcudf needs to have a SizeType DataType singleton
                    plc.DataType(plc.TypeId.INT32),
                )
            ).as_py()
        return 0


class NamedColumn(Column):
    """A column with a name."""

    name: str

    def __init__(
        self,
        column: plc.Column,
        name: str,
        *,
        is_sorted: plc.types.Sorted = plc.types.Sorted.NO,
        order: plc.types.Order = plc.types.Order.ASCENDING,
        null_order: plc.types.NullOrder = plc.types.NullOrder.BEFORE,
    ) -> None:
        super().__init__(
            column, is_sorted=is_sorted, order=order, null_order=null_order
        )
        self.name = name

    def copy(self, *, new_name: str | None = None) -> Self:
        """
        A shallow copy of the column.

        Parameters
        ----------
        new_name
            Optional new name for the copied column.

        Returns
        -------
        New column sharing data with self.
        """
        return type(self)(
            self.obj,
            self.name if new_name is None else new_name,
            is_sorted=self.is_sorted,
            order=self.order,
            null_order=self.null_order,
        )

    def mask_nans(self) -> Self:
        """Return a shallow copy of self with nans masked out."""
        # Annoying, the inheritance is not right (can't call the
        # super-type mask_nans), but will sort that by refactoring
        # later.
        if plc.traits.is_floating_point(self.obj.type()):
            old_count = self.obj.null_count()
            mask, new_count = plc.transform.nans_to_nulls(self.obj)
            result = type(self)(self.obj.with_mask(mask, new_count), self.name)
            if old_count == new_count:
                return result.sorted_like(self)
            return result
        return self.copy()
