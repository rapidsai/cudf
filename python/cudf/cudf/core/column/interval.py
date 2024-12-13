# Copyright (c) 2018-2024, NVIDIA CORPORATION.
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import pyarrow as pa

import cudf
from cudf.core.column.column import as_column
from cudf.core.column.struct import StructColumn
from cudf.core.dtypes import IntervalDtype

if TYPE_CHECKING:
    from typing_extensions import Self

    from cudf.core.buffer import Buffer
    from cudf.core.column import ColumnBase


class IntervalColumn(StructColumn):
    def __init__(
        self,
        data: None,
        size: int,
        dtype: IntervalDtype,
        mask: Buffer | None = None,
        offset: int = 0,
        null_count: int | None = None,
        children: tuple[ColumnBase, ColumnBase] = (),  # type: ignore[assignment]
    ):
        if len(children) != 2:
            raise ValueError(
                "children must be a tuple of two columns (left edges, right edges)."
            )
        super().__init__(
            data=data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    @staticmethod
    def _validate_dtype_instance(dtype: IntervalDtype) -> IntervalDtype:
        if not isinstance(dtype, IntervalDtype):
            raise ValueError("dtype must be a IntervalDtype.")
        return dtype

    @classmethod
    def from_arrow(cls, data: pa.Array) -> Self:
        new_col = super().from_arrow(data.storage)
        size = len(data)
        dtype = IntervalDtype.from_arrow(data.type)
        mask = data.buffers()[0]
        if mask is not None:
            mask = cudf.utils.utils.pa_mask_buffer_to_mask(mask, len(data))

        offset = data.offset
        null_count = data.null_count
        children = new_col.children

        return cls(
            data=None,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,  # type: ignore[arg-type]
        )

    def to_arrow(self) -> pa.Array:
        typ = self.dtype.to_arrow()
        struct_arrow = super().to_arrow()
        if len(struct_arrow) == 0:
            # struct arrow is pa.struct array with null children types
            # we need to make sure its children have non-null type
            struct_arrow = pa.array([], typ.storage_type)
        return pa.ExtensionArray.from_storage(typ, struct_arrow)

    @classmethod
    def from_struct_column(
        cls,
        struct_column: StructColumn,
        closed: Literal["left", "right", "both", "neither"] = "right",
    ) -> Self:
        first_field_name = next(iter(struct_column.dtype.fields.keys()))
        return cls(
            data=None,
            size=struct_column.size,
            dtype=IntervalDtype(
                struct_column.dtype.fields[first_field_name], closed
            ),
            mask=struct_column.base_mask,
            offset=struct_column.offset,
            null_count=struct_column.null_count,
            children=struct_column.base_children,  # type: ignore[arg-type]
        )

    def copy(self, deep: bool = True) -> Self:
        struct_copy = super().copy(deep=deep)
        return IntervalColumn(  # type: ignore[return-value]
            data=None,
            size=struct_copy.size,
            dtype=IntervalDtype(
                struct_copy.dtype.fields["left"], self.dtype.closed
            ),
            mask=struct_copy.base_mask,
            offset=struct_copy.offset,
            null_count=struct_copy.null_count,
            children=struct_copy.base_children,  # type: ignore[arg-type]
        )

    @property
    def is_empty(self) -> ColumnBase:
        left_equals_right = (self.right == self.left).fillna(False)
        not_closed_both = as_column(
            self.dtype.closed != "both", length=len(self)
        )
        return left_equals_right & not_closed_both

    @property
    def is_non_overlapping_monotonic(self) -> bool:
        raise NotImplementedError(
            "is_overlapping is currently not implemented."
        )

    @property
    def is_overlapping(self) -> bool:
        raise NotImplementedError(
            "is_overlapping is currently not implemented."
        )

    @property
    def length(self) -> ColumnBase:
        return self.right - self.left

    @property
    def left(self) -> ColumnBase:
        return self.children[0]

    @property
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
        return IntervalColumn(  # type: ignore[return-value]
            data=None,
            size=self.size,
            dtype=IntervalDtype(self.dtype.fields["left"], closed),
            mask=self.base_mask,
            offset=self.offset,
            null_count=self.null_count,
            children=self.base_children,  # type: ignore[arg-type]
        )

    def as_interval_column(self, dtype: IntervalDtype) -> Self:  # type: ignore[override]
        if isinstance(dtype, IntervalDtype):
            return IntervalColumn(  # type: ignore[return-value]
                data=None,
                size=self.size,
                dtype=dtype,
                mask=self.mask,
                offset=self.offset,
                null_count=self.null_count,
                children=tuple(  # type: ignore[arg-type]
                    child.astype(dtype.subtype) for child in self.children
                ),
            )
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
        if nullable:
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif arrow_type:
            raise NotImplementedError(f"{arrow_type=} is not implemented.")

        pd_type = self.dtype.to_pandas()
        return pd.Index(pd_type.__from_arrow__(self.to_arrow()), dtype=pd_type)

    def element_indexing(self, index: int):
        result = super().element_indexing(index)
        if cudf.get_option("mode.pandas_compatible"):
            return pd.Interval(**result, closed=self.dtype.closed)
        return result
