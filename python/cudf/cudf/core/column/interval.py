# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import pandas as pd
import pyarrow as pa

import pylibcudf as plc

from cudf.core._internals import binaryop
from cudf.core.column.column import (
    ColumnBase,
    as_column,
    dtype_from_pylibcudf_column,
)
from cudf.core.dtypes import IntervalDtype
from cudf.utils.dtypes import get_dtype_of_same_kind

if TYPE_CHECKING:
    from cudf._typing import ColumnBinaryOperand, DtypeObj
    from cudf.core.buffer import Buffer


class IntervalColumn(ColumnBase):
    @classmethod
    def from_arrow(cls, array: pa.Array | pa.ChunkedArray) -> Self:
        if not isinstance(array, pa.ExtensionArray):
            raise ValueError("Expected ExtensionArray for interval data")
        new_col = super().from_arrow(array.storage)
        return ColumnBase.create(
            new_col.plc_column, IntervalDtype.from_arrow(array.type)
        )  # type: ignore[return-value]

    @functools.cached_property
    def closed(self) -> Literal["left", "right", "neither", "both"]:
        if isinstance(self.dtype, IntervalDtype):
            return self.dtype.closed
        else:
            return cast("pd.ArrowDtype", self.dtype).pyarrow_dtype.closed

    def to_arrow(self) -> pa.Array:
        pa_array = super().to_arrow()
        pa_type = (
            self.dtype.to_arrow()
            if isinstance(self.dtype, IntervalDtype)
            else cast("pd.ArrowDtype", self.dtype).pyarrow_dtype
        )
        if pa.types.is_null(pa_array.type):
            # from _handle_nulls
            # pa.ExtensionArray.from_storage cannot handle null arrays
            pa_array = pa.array([], type=pa_type.storage_type)
        return pa.ExtensionArray.from_storage(pa_type, pa_array)

    @classmethod
    def _deserialize_plc_column(
        cls,
        header: dict,
        dtype: DtypeObj,
        data: Buffer | None,
        mask: Buffer | None,
        children: list[plc.Column],
    ) -> plc.Column:
        """Construct plc.Column using STRUCT type for interval columns."""
        offset = header.get("offset", 0)
        if mask is None:
            null_count = 0
        else:
            null_count = plc.null_mask.null_count(
                mask, offset, header["size"] + offset
            )

        plc_type = plc.DataType(plc.TypeId.STRUCT)
        return plc.Column(
            plc_type,
            header["size"],
            data,
            mask,
            null_count,
            offset,
            children,
            validate=False,
        )

    def copy(self, deep: bool = True) -> Self:
        plc_col = self.plc_column
        if deep:
            plc_col = plc_col.copy()
        return ColumnBase.create(plc_col, self.dtype)  # type: ignore[return-value]

    @functools.cached_property
    def is_empty(self) -> ColumnBase:
        left_equals_right = (self.right == self.left).fillna(False)
        not_closed_both = as_column(
            self.dtype.closed != "both",  # type: ignore[union-attr]
            length=len(self),
        )
        return left_equals_right & not_closed_both

    @functools.cached_property
    def is_non_overlapping_monotonic(self) -> bool:
        raise NotImplementedError(
            "is_overlapping is currently not implemented."
        )

    @functools.cached_property
    def is_overlapping(self) -> bool:
        raise NotImplementedError(
            "is_overlapping is currently not implemented."
        )

    @functools.cached_property
    def length(self) -> ColumnBase:
        return self.right - self.left

    @property
    def left(self) -> ColumnBase:
        return ColumnBase.create(
            self.plc_column.children()[0],
            self.dtype.subtype,  # type: ignore[union-attr]
        )

    @functools.cached_property
    def mid(self) -> ColumnBase:
        try:
            return 0.5 * (self.left + self.right)
        except TypeError:
            # datetime safe version
            return self.left + 0.5 * self.length

    @property
    def right(self) -> ColumnBase:
        return ColumnBase.create(
            self.plc_column.children()[1],
            self.dtype.subtype,  # type: ignore[union-attr]
        )

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        raise NotImplementedError(
            "Intervals are not yet supported via `__cuda_array_interface__`"
        )

    def overlaps(other) -> ColumnBase:
        raise NotImplementedError("overlaps is not currently implemented.")

    def set_closed(
        self, closed: Literal["left", "right", "both", "neither"]
    ) -> Self:
        return ColumnBase.create(  # type: ignore[return-value]
            self.plc_column,
            IntervalDtype(self.dtype.subtype, closed),  # type: ignore[union-attr]
        )

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)
        if not isinstance(other, type(self)):
            return NotImplemented
        if op == "NULL_EQUALS":
            lefts_equal = self.left._binaryop(other.left, "NULL_EQUALS")
            rights_equal = self.right._binaryop(other.right, "NULL_EQUALS")
            return binaryop.binaryop(
                lefts_equal,
                rights_equal,
                "__and__",
                get_dtype_of_same_kind(self.dtype, lefts_equal.dtype),
            )
        else:
            raise TypeError(f"{op} not supported with {type(other).__name__}")

    def as_interval_column(self, dtype: IntervalDtype) -> Self:
        if not isinstance(dtype, IntervalDtype):
            raise ValueError("dtype must be IntervalDtype")

        # If subtype is changing, cast children to match new subtype
        if dtype.subtype != self.dtype.subtype:  # type: ignore[union-attr]
            new_children = tuple(
                ColumnBase.create(
                    child, dtype_from_pylibcudf_column(child)
                ).astype(dtype.subtype)
                for child in self.plc_column.children()
            )
            # Reconstruct plc_column with cast children
            plc_column = plc.Column(
                plc.DataType(plc.TypeId.STRUCT),
                self.plc_column.size(),
                self.plc_column.data(),
                self.plc_column.null_mask(),
                self.plc_column.null_count(),
                self.plc_column.offset(),
                [child.plc_column for child in new_children],
            )
        else:
            plc_column = self.plc_column

        return ColumnBase.create(plc_column, dtype)  # type: ignore[return-value]

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        # Note: This does not handle null values in the interval column.
        # However, this exact sequence (calling __from_arrow__ on the output of
        # self.to_arrow) is currently the best known way to convert interval
        # types into pandas (trying to convert the underlying numerical columns
        # directly is problematic), so we're stuck with this for now.
        if nullable or isinstance(self.dtype, pd.ArrowDtype):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif arrow_type:
            raise NotImplementedError(f"{arrow_type=} is not implemented.")

        pd_type = self.dtype.to_pandas()  # type: ignore[union-attr]
        return pd.Index(pd_type.__from_arrow__(self.to_arrow()), dtype=pd_type)

    def element_indexing(self, index: int) -> pd.Interval | None:
        result = super().element_indexing(index)
        if isinstance(result, dict):
            return pd.Interval(**result, closed=self.closed)
        return result
