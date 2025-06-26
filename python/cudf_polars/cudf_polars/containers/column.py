# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A column, with some properties."""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING

import polars as pl
from polars.exceptions import InvalidOperationError

import pylibcudf as plc
from pylibcudf.strings.convert.convert_floats import from_floats, is_float, to_floats
from pylibcudf.strings.convert.convert_integers import (
    from_integers,
    is_integer,
    to_integers,
)
from pylibcudf.traits import is_floating_point

from cudf_polars.containers import DataType
from cudf_polars.utils import conversion
from cudf_polars.utils.dtypes import is_order_preserving_cast

if TYPE_CHECKING:
    from typing_extensions import Self

    from cudf_polars.typing import (
        ColumnHeader,
        ColumnOptions,
        DeserializedColumnOptions,
        Slice,
    )

__all__: list[str] = ["Column"]


def _dtype_short_repr_to_dtype(dtype_str: str) -> pl.DataType:
    """Convert a Polars dtype short repr to a Polars dtype."""
    # limitations of dtype_short_repr_to_dtype described in
    # py-polars/polars/datatypes/convert.py#L299
    if dtype_str.startswith("list["):
        stripped = dtype_str.removeprefix("list[").removesuffix("]")
        return pl.List(_dtype_short_repr_to_dtype(stripped))
    pl_type = pl.datatypes.convert.dtype_short_repr_to_dtype(dtype_str)
    if pl_type is None:
        raise ValueError(f"{dtype_str} was not able to be parsed by Polars.")
    return pl_type() if inspect.isclass(pl_type) else pl_type


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
    dtype: DataType

    def __init__(
        self,
        column: plc.Column,
        dtype: DataType,
        *,
        is_sorted: plc.types.Sorted = plc.types.Sorted.NO,
        order: plc.types.Order = plc.types.Order.ASCENDING,
        null_order: plc.types.NullOrder = plc.types.NullOrder.BEFORE,
        name: str | None = None,
    ):
        self.obj = column
        self.is_scalar = self.size == 1
        self.name = name
        self.dtype = dtype
        self.set_sorted(is_sorted=is_sorted, order=order, null_order=null_order)

    @classmethod
    def deserialize(
        cls, header: ColumnHeader, frames: tuple[memoryview, plc.gpumemoryview]
    ) -> Self:
        """
        Create a Column from a serialized representation returned by `.serialize()`.

        Parameters
        ----------
        header
            The (unpickled) metadata required to reconstruct the object.
        frames
            Two-tuple of frames (a memoryview and a gpumemoryview).

        Returns
        -------
        Column
            The deserialized Column.
        """
        packed_metadata, packed_gpu_data = frames
        (plc_column,) = plc.contiguous_split.unpack_from_memoryviews(
            packed_metadata, packed_gpu_data
        ).columns()
        return cls(plc_column, **cls.deserialize_ctor_kwargs(header["column_kwargs"]))

    @staticmethod
    def deserialize_ctor_kwargs(
        column_kwargs: ColumnOptions,
    ) -> DeserializedColumnOptions:
        """Deserialize the constructor kwargs for a Column."""
        dtype = DataType(  # pragma: no cover
            _dtype_short_repr_to_dtype(column_kwargs["dtype"])
        )
        return {
            "is_sorted": column_kwargs["is_sorted"],
            "order": column_kwargs["order"],
            "null_order": column_kwargs["null_order"],
            "name": column_kwargs["name"],
            "dtype": dtype,
        }

    def serialize(
        self,
    ) -> tuple[ColumnHeader, tuple[memoryview, plc.gpumemoryview]]:
        """
        Serialize the Column into header and frames.

        Follows the Dask serialization scheme with a picklable header (dict) and
        a tuple of frames (in this case a contiguous host and device buffer).

        To enable dask support, dask serializers must be registered

            >>> from cudf_polars.experimental.dask_serialize import register
            >>> register()

        Returns
        -------
        header
            A dict containing any picklable metadata required to reconstruct the object.
        frames
            Two-tuple of frames suitable for passing to `plc.contiguous_split.unpack_from_memoryviews`
        """
        packed = plc.contiguous_split.pack(plc.Table([self.obj]))
        header: ColumnHeader = {
            "column_kwargs": self.serialize_ctor_kwargs(),
            "frame_count": 2,
        }
        return header, packed.release()

    def serialize_ctor_kwargs(self) -> ColumnOptions:
        """Serialize the constructor kwargs for self."""
        return {
            "is_sorted": self.is_sorted,
            "order": self.order,
            "null_order": self.null_order,
            "name": self.name,
            "dtype": pl.polars.dtype_str_repr(self.dtype.polars),
        }

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
            raise ValueError(f"Cannot convert a column of length {self.size} to scalar")
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
            dtype=self.dtype,
            is_sorted=like.is_sorted,
            order=like.order,
            null_order=like.null_order,
        )

    def check_sorted(
        self,
        *,
        order: plc.types.Order,
        null_order: plc.types.NullOrder,
    ) -> bool:
        """
        Check if the column is sorted.

        Parameters
        ----------
        order
            The requested sort order.
        null_order
            Where nulls sort to.

        Returns
        -------
        True if the column is sorted, false otherwise.

        Notes
        -----
        If the sortedness flag is not set, this launches a kernel to
        check sortedness.
        """
        if self.size <= 1 or self.size == self.null_count:
            return True
        if self.is_sorted == plc.types.Sorted.YES:
            return self.order == order and (
                self.null_count == 0 or self.null_order == null_order
            )
        if plc.sorting.is_sorted(plc.Table([self.obj]), [order], [null_order]):
            self.sorted = plc.types.Sorted.YES
            self.order = order
            self.null_order = null_order
            return True
        return False

    def astype(self, dtype: DataType) -> Column:
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
        plc_dtype = dtype.plc
        if self.obj.type() == plc_dtype:
            return self

        if (
            plc_dtype.id() == plc.TypeId.STRING
            or self.obj.type().id() == plc.TypeId.STRING
        ):
            return Column(self._handle_string_cast(plc_dtype), dtype=dtype)
        else:
            result = Column(plc.unary.cast(self.obj, plc_dtype), dtype=dtype)
            if is_order_preserving_cast(self.obj.type(), plc_dtype):
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
                if not plc.reduce.reduce(
                    floats,
                    plc.aggregation.all(),
                    plc.DataType(plc.TypeId.BOOL8),
                ).to_py():
                    raise InvalidOperationError("Conversion from `str` failed.")
                return to_floats(self.obj, dtype)
            else:
                integers = is_integer(self.obj)
                if not plc.reduce.reduce(
                    integers,
                    plc.aggregation.all(),
                    plc.DataType(plc.TypeId.BOOL8),
                ).to_py():
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
        if self.size <= 1:
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
            dtype=self.dtype,
        )

    def mask_nans(self) -> Self:
        """Return a shallow copy of self with nans masked out."""
        if plc.traits.is_floating_point(self.obj.type()):
            old_count = self.null_count
            mask, new_count = plc.transform.nans_to_nulls(self.obj)
            result = type(self)(self.obj.with_mask(mask, new_count), self.dtype)
            if old_count == new_count:
                return result.sorted_like(self)
            return result
        return self.copy()

    @functools.cached_property
    def nan_count(self) -> int:
        """Return the number of NaN values in the column."""
        if plc.traits.is_floating_point(self.obj.type()):
            return plc.reduce.reduce(
                plc.unary.is_nan(self.obj),
                plc.aggregation.sum(),
                plc.types.SIZE_TYPE,
            ).to_py()
        return 0

    @property
    def size(self) -> int:
        """Return the size of the column."""
        return self.obj.size()

    @property
    def null_count(self) -> int:
        """Return the number of Null values in the column."""
        return self.obj.null_count()

    def slice(self, zlice: Slice | None) -> Self:
        """
        Slice a column.

        Parameters
        ----------
        zlice
            optional, tuple of start and length, negative values of start
            treated as for python indexing. If not provided, returns self.

        Returns
        -------
        New column (if zlice is not None) otherwise self (if it is)
        """
        if zlice is None:
            return self
        (table,) = plc.copying.slice(
            plc.Table([self.obj]),
            conversion.from_polars_slice(zlice, num_rows=self.size),
        )
        (column,) = table.columns()
        return type(self)(column, name=self.name, dtype=self.dtype).sorted_like(self)
