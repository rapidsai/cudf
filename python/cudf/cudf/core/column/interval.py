# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import pylibcudf as plc

import cudf
from cudf.core.column.column import ColumnBase, _handle_nulls, as_column
from cudf.core.dtype.validators import is_dtype_obj_interval
from cudf.core.dtypes import IntervalDtype, _dtype_to_metadata
from cudf.utils.scalar import maybe_nested_pa_scalar_to_py

if TYPE_CHECKING:
    from cudf._typing import DtypeObj
    from cudf.core.buffer import Buffer


class IntervalColumn(ColumnBase):
    _VALID_PLC_TYPES = {plc.TypeId.STRUCT}

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
        if not is_dtype_obj_interval(dtype):
            raise ValueError("dtype must be a IntervalDtype.")

        # Validate that children dtypes are compatible with target subtype
        for i, child in enumerate(plc_column.children()):
            try:
                ColumnBase._validate_dtype_recursively(child, dtype.subtype)
            except ValueError as e:
                raise ValueError(
                    f"{'Right' if i else 'Left'} interval bound validation failed: {e}"
                ) from e

        return plc_column, dtype

    def _with_type_metadata(self, dtype: DtypeObj) -> ColumnBase:
        """
        Apply IntervalDtype metadata to this column.

        Creates new children with the subtype metadata applied and
        reconstructs the plc.Column.
        """
        if isinstance(dtype, IntervalDtype):
            new_children = tuple(
                ColumnBase.from_pylibcudf(child).astype(dtype.subtype)
                for child in self.plc_column.children()
            )
            new_plc_column = plc.Column(
                plc.DataType(plc.TypeId.STRUCT),
                self.plc_column.size(),
                self.plc_column.data(),
                self.plc_column.null_mask(),
                self.plc_column.null_count(),
                self.plc_column.offset(),
                [child.plc_column for child in new_children],
            )
            return type(self)._from_preprocessed(
                plc_column=new_plc_column,
                dtype=dtype,
            )
        # For pandas dtypes, store them directly in the column's dtype property
        elif isinstance(dtype, pd.ArrowDtype) and isinstance(
            dtype.pyarrow_dtype, ArrowIntervalType
        ):
            self._dtype = dtype

        return self

    @classmethod
    def from_arrow(cls, array: pa.Array | pa.ChunkedArray) -> Self:
        if not isinstance(array, pa.ExtensionArray):
            raise ValueError("Expected ExtensionArray for interval data")
        new_col = super().from_arrow(array.storage)
        return ColumnBase.create(
            new_col.plc_column, IntervalDtype.from_arrow(array.type)
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

    def as_interval_column(self, dtype: IntervalDtype) -> Self:
        if not isinstance(dtype, IntervalDtype):
            raise ValueError("dtype must be IntervalDtype")

        # If subtype is changing, cast children to match new subtype
        if dtype.subtype != self.dtype.subtype:  # type: ignore[union-attr]
            new_children = tuple(
                ColumnBase.from_pylibcudf(child).astype(dtype.subtype)
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

    def element_indexing(
        self, index: int
    ) -> pd.Interval | dict[Any, Any] | None:
        result = super().element_indexing(index)
        if isinstance(result, pa.Scalar):
            py_element = maybe_nested_pa_scalar_to_py(result)
            result = self.dtype._recursively_replace_fields(py_element)  # type: ignore[union-attr]
        if isinstance(result, dict) and cudf.get_option(
            "mode.pandas_compatible"
        ):
            return pd.Interval(**result, closed=self.dtype.closed)  # type: ignore[union-attr]
        return result
