# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import pyarrow as pa
from typing_extensions import Self

import pylibcudf as plc

import cudf
from cudf.core.column.column import _handle_nulls, as_column
from cudf.core.column.struct import StructColumn
from cudf.core.dtypes import IntervalDtype, _dtype_to_metadata
from cudf.utils.dtypes import is_dtype_obj_interval

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase


class IntervalColumn(StructColumn):
    @classmethod
    def _validate_args(  # type: ignore[override]
        cls, plc_column: plc.Column, dtype: IntervalDtype
    ) -> tuple[plc.Column, IntervalDtype]:
        # Validate plc_column TypeId - IntervalColumn uses STRUCT type
        if not (
            isinstance(plc_column, plc.Column)
            and plc_column.type().id() == plc.TypeId.STRUCT
        ):
            raise ValueError(
                "plc_column must be a pylibcudf.Column with TypeId STRUCT"
            )
        if plc_column.num_children() != 2:
            raise ValueError(
                "plc_column must have two children (left edges, right edges)."
            )
        if (
            not cudf.get_option("mode.pandas_compatible")
            and not isinstance(dtype, IntervalDtype)
        ) or (
            cudf.get_option("mode.pandas_compatible")
            and not is_dtype_obj_interval(dtype)
        ):
            raise ValueError("dtype must be a IntervalDtype.")
        return plc_column, dtype

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
        struct_arrow = self.plc_column.to_arrow(
            metadata=_dtype_to_metadata(self.dtype)
        )
        possibly_null_struct_arrow = _handle_nulls(struct_arrow)

        # Disable null handling for all null arrays because those cannot be
        # passed to from_storage below, in that case we leave the struct
        # structure in place.
        if not isinstance(possibly_null_struct_arrow, pa.lib.NullArray):
            struct_arrow = possibly_null_struct_arrow

        if len(struct_arrow) == 0:
            # struct arrow is pa.struct array with null children types
            # we need to make sure its children have non-null type
            struct_arrow = pa.array([], typ.storage_type)
        return pa.ExtensionArray.from_storage(typ, struct_arrow)

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> Self:
        """Override deserialize to handle interval-specific dtype conversion."""

        def unpack(header: dict, frames: list) -> tuple[Any, list]:
            count = header["frame_count"]
            obj = cls.device_deserialize(header, frames[:count])
            return obj, frames[count:]

        assert header["frame_count"] == len(frames), (
            f"Deserialization expected {header['frame_count']} frames, "
            f"but received {len(frames)}"
        )
        if header["dtype-is-cudf-serialized"]:
            dtype, frames = unpack(header["dtype"], frames)
        else:
            try:
                import numpy as np

                dtype = np.dtype(header["dtype"])
            except TypeError:
                import pickle

                dtype = pickle.loads(header["dtype"])
        if "data" in header:
            data, frames = unpack(header["data"], frames)
        else:
            data = None
        if "mask" in header:
            mask, frames = unpack(header["mask"], frames)
        else:
            mask = None
        children = []
        if "subheaders" in header:
            for h in header["subheaders"]:
                child, frames = unpack(h, frames)
                children.append(child)
        assert len(frames) == 0, "Deserialization did not consume all frames"
        offset = header.get("offset", 0)
        if mask is None:
            null_count = 0
        else:
            null_count = plc.null_mask.null_count(
                mask, offset, header["size"] + offset
            )

        # Interval-specific deserialization:
        # Use STRUCT type instead of interval type for plc_column construction
        plc_type = plc.DataType(plc.TypeId.STRUCT)
        plc_column = plc.Column(
            plc_type,
            header["size"],
            data,
            mask,
            null_count,
            offset,
            [child.plc_column for child in children],
            validate=False,
        )
        from typing import cast

        result = cls.from_pylibcudf(plc_column)._with_type_metadata(dtype)
        return cast(Self, result)

    def copy(self, deep: bool = True) -> Self:
        return super().copy(deep=deep)._with_type_metadata(self.dtype)  # type: ignore[return-value]

    def reduce(self, reduction_op: str, **kwargs: Any) -> Any:
        """Override reduce to preserve IntervalDtype metadata."""
        result_col_dtype = self._reduction_result_dtype(reduction_op)

        # check empty case
        if len(self) <= self.null_count:
            if reduction_op == "sum" or reduction_op == "sum_of_squares":
                return result_col_dtype.type(0)
            if reduction_op == "product":
                return result_col_dtype.type(1)
            if reduction_op == "any":
                return False

            from cudf.utils.dtypes import _get_nan_for_dtype

            return _get_nan_for_dtype(result_col_dtype)

        from cudf.core._internals import aggregation
        from cudf.utils.dtypes import dtype_to_pylibcudf_type

        with self.access(mode="read", scope="internal"):
            plc_scalar = plc.reduce.reduce(
                self.plc_column,
                aggregation.make_aggregation(reduction_op, kwargs).plc_obj,
                dtype_to_pylibcudf_type(result_col_dtype),
            )
            result_col = type(self).from_pylibcudf(
                plc.Column.from_scalar(plc_scalar, 1)
            )
            # Apply interval dtype metadata to the result
            if isinstance(result_col_dtype, IntervalDtype):
                result_col = result_col._with_type_metadata(result_col_dtype)
        return result_col.element_indexing(0)

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
