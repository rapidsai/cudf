# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A column, with some properties."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from polars.exceptions import InvalidOperationError

import pylibcudf as plc
from pylibcudf.strings.convert.convert_floats import from_floats, is_float, to_floats
from pylibcudf.strings.convert.convert_integers import (
    from_integers,
    is_integer,
    to_integers,
)
from pylibcudf.traits import is_floating_point

from cudf_polars.utils.dtypes import is_order_preserving_cast

if TYPE_CHECKING:
    from typing_extensions import Self

    import polars as pl

__all__: list[str] = ["Column"]


class Column:
    """An immutable column with sortedness metadata."""

    obj: plc.Column
    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder
    is_scalar: bool
    # Optional name, only ever set by evaluation of NamedExpr nodes
    # The internal evaluation should not care about the name.
    name: str | None

    def __init__(
        self,
        column: plc.Column,
        *,
        is_sorted: plc.types.Sorted = plc.types.Sorted.NO,
        order: plc.types.Order = plc.types.Order.ASCENDING,
        null_order: plc.types.NullOrder = plc.types.NullOrder.BEFORE,
        name: str | None = None,
    ):
        self.obj = column
        self.is_scalar = self.obj.size() == 1
        self.name = name
        self.set_sorted(is_sorted=is_sorted, order=order, null_order=null_order)

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

    def rename(self, name: str | None, /) -> Self:
        """
        Return a shallow copy with a new name.

        Parameters
        ----------
        name
            New name

        Returns
        -------
        Shallow copy of self with new name set.
        """
        new = self.copy()
        new.name = name
        return new

    def sorted_like(self, like: Column, /) -> Self:
        """
        Return a shallow copy with sortedness from like.

        Parameters
        ----------
        like
            The column to copy sortedness metadata from.

        Returns
        -------
        Shallow copy of self with metadata set.

        See Also
        --------
        set_sorted, copy_metadata
        """
        return type(self)(
            self.obj,
            name=self.name,
            is_sorted=like.is_sorted,
            order=like.order,
            null_order=like.null_order,
        )

    def astype(self, dtype: plc.DataType) -> Column:
        """
        Cast the column to as the requested dtype.

        Parameters
        ----------
        dtype
            Datatype to cast to.

        Returns
        -------
        Column of requested type.

        Raises
        ------
        RuntimeError
            If the cast is unsupported.

        Notes
        -----
        This only produces a copy if the requested dtype doesn't match
        the current one.
        """
        if self.obj.type() == dtype:
            return self

        if dtype.id() == plc.TypeId.STRING or self.obj.type().id() == plc.TypeId.STRING:
            return Column(self._handle_string_cast(dtype))
        else:
            result = Column(plc.unary.cast(self.obj, dtype))
            if is_order_preserving_cast(self.obj.type(), dtype):
                return result.sorted_like(self)
            return result

    def _handle_string_cast(self, dtype: plc.DataType) -> plc.Column:
        if dtype.id() == plc.TypeId.STRING:
            if is_floating_point(self.obj.type()):
                return from_floats(self.obj)
            else:
                return from_integers(self.obj)
        else:
            if is_floating_point(dtype):
                floats = is_float(self.obj)
                if not plc.interop.to_arrow(
                    plc.reduce.reduce(
                        floats,
                        plc.aggregation.all(),
                        plc.DataType(plc.TypeId.BOOL8),
                    )
                ).as_py():
                    raise InvalidOperationError("Conversion from `str` failed.")
                return to_floats(self.obj, dtype)
            else:
                integers = is_integer(self.obj)
                if not plc.interop.to_arrow(
                    plc.reduce.reduce(
                        integers,
                        plc.aggregation.all(),
                        plc.DataType(plc.TypeId.BOOL8),
                    )
                ).as_py():
                    raise InvalidOperationError("Conversion from `str` failed.")
                return to_integers(self.obj, dtype)

    def copy_metadata(self, from_: pl.Series, /) -> Self:
        """
        Copy metadata from a host series onto self.

        Parameters
        ----------
        from_
            Polars series to copy metadata from

        Returns
        -------
        Self with metadata set.

        See Also
        --------
        set_sorted, sorted_like
        """
        self.name = from_.name
        if len(from_) <= 1:
            return self
        ascending = from_.flags["SORTED_ASC"]
        descending = from_.flags["SORTED_DESC"]
        if ascending or descending:
            has_null_first = from_.item(0) is None
            has_null_last = from_.item(-1) is None
            order = (
                plc.types.Order.ASCENDING if ascending else plc.types.Order.DESCENDING
            )
            null_order = plc.types.NullOrder.BEFORE
            if (descending and has_null_first) or (ascending and has_null_last):
                null_order = plc.types.NullOrder.AFTER
            return self.set_sorted(
                is_sorted=plc.types.Sorted.YES,
                order=order,
                null_order=null_order,
            )
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
            name=self.name,
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
                    plc.types.SIZE_TYPE,
                )
            ).as_py()
        return 0
