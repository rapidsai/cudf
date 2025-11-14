# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import pyarrow as pa

import cudf
from cudf.core.column.column import as_column
from cudf.core.column.struct import StructColumn
from cudf.core.dtypes import IntervalDtype
from cudf.utils.dtypes import is_dtype_obj_interval

if TYPE_CHECKING:
    from typing_extensions import Self

    import pylibcudf as plc

    from cudf.core.column import ColumnBase


class IntervalColumn(StructColumn):
    def __init__(
        self,
        plc_column: plc.Column,
        size: int,
        dtype: IntervalDtype,
        offset: int,
        null_count: int,
        exposed: bool,
    ) -> None:
        if plc_column.num_children() != 2:
            raise ValueError(
                "plc_column must have two children (left edges, right edges)."
            )
        super().__init__(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )

    @staticmethod
    def _validate_dtype_instance(dtype: IntervalDtype) -> IntervalDtype:
        if (
            not cudf.get_option("mode.pandas_compatible")
            and not isinstance(dtype, IntervalDtype)
        ) or (
            cudf.get_option("mode.pandas_compatible")
            and not is_dtype_obj_interval(dtype)
        ):
            raise ValueError("dtype must be a IntervalDtype.")
        return dtype

    @classmethod
    def from_arrow(cls, array: pa.Array | pa.ChunkedArray) -> Self:
        if not isinstance(array, pa.ExtensionArray):
            raise ValueError("Expected ExtensionArray for interval data")
        new_col = super().from_arrow(array.storage)
        return new_col._with_type_metadata(
            IntervalDtype.from_arrow(array.type)
        )  # type: ignore[return-value]

    def to_arrow(self) -> pa.Array:
        typ = self.dtype.to_arrow()  # type: ignore[union-attr]
        struct_arrow = super().to_arrow()
        if len(struct_arrow) == 0:
            # struct arrow is pa.struct array with null children types
            # we need to make sure its children have non-null type
            struct_arrow = pa.array([], typ.storage_type)
        return pa.ExtensionArray.from_storage(typ, struct_arrow)

    def copy(self, deep: bool = True) -> Self:
        return super().copy(deep=deep)._with_type_metadata(self.dtype)  # type: ignore[return-value]

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
        return self.children[0]

    @functools.cached_property
    def mid(self) -> ColumnBase:
        try:
            return 0.5 * (self.left + self.right)
        except TypeError:
            # datetime safe version
            return self.left + 0.5 * self.length

    @property
    def right(self) -> ColumnBase:
        return self.children[1]

    def overlaps(other) -> ColumnBase:
        raise NotImplementedError("overlaps is not currently implemented.")

    def set_closed(
        self, closed: Literal["left", "right", "both", "neither"]
    ) -> Self:
        return self._with_type_metadata(  # type: ignore[return-value]
            IntervalDtype(self.dtype.subtype, closed)  # type: ignore[union-attr]
        )

    def as_interval_column(self, dtype: IntervalDtype) -> Self:
        if isinstance(dtype, IntervalDtype):
            return self._with_type_metadata(dtype)  # type: ignore[return-value]
        else:
            raise ValueError("dtype must be IntervalDtype")

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
        if nullable or (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(self.dtype, pd.ArrowDtype)
        ):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif arrow_type:
            raise NotImplementedError(f"{arrow_type=} is not implemented.")

        pd_type = self.dtype.to_pandas()  # type: ignore[union-attr]
        return pd.Index(pd_type.__from_arrow__(self.to_arrow()), dtype=pd_type)

    def element_indexing(
        self, index: int
    ) -> pd.Interval | dict[Any, Any] | None:
        result = super().element_indexing(index)
        if isinstance(result, dict) and cudf.get_option(
            "mode.pandas_compatible"
        ):
            return pd.Interval(**result, closed=self.dtype.closed)  # type: ignore[union-attr]
        return result
