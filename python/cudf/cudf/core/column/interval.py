# Copyright (c) 2018-2024, NVIDIA CORPORATION.
import pandas as pd
import pyarrow as pa

import cudf
from cudf.core.column import StructColumn
from cudf.core.dtypes import IntervalDtype


class IntervalColumn(StructColumn):
    def __init__(
        self,
        dtype,
        mask=None,
        size=None,
        offset=0,
        null_count=None,
        children=(),
    ):
        super().__init__(
            data=None,
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    @classmethod
    def from_arrow(cls, data):
        new_col = super().from_arrow(data.storage)
        size = len(data)
        dtype = IntervalDtype.from_arrow(data.type)
        mask = data.buffers()[0]
        if mask is not None:
            mask = cudf.utils.utils.pa_mask_buffer_to_mask(mask, len(data))

        offset = data.offset
        null_count = data.null_count
        children = new_col.children

        return IntervalColumn(
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    def to_arrow(self):
        typ = self.dtype.to_arrow()
        struct_arrow = super().to_arrow()
        if len(struct_arrow) == 0:
            # struct arrow is pa.struct array with null children types
            # we need to make sure its children have non-null type
            struct_arrow = pa.array([], typ.storage_type)
        return pa.ExtensionArray.from_storage(typ, struct_arrow)

    @classmethod
    def from_struct_column(cls, struct_column: StructColumn, closed="right"):
        first_field_name = next(iter(struct_column.dtype.fields.keys()))
        return IntervalColumn(
            size=struct_column.size,
            dtype=IntervalDtype(
                struct_column.dtype.fields[first_field_name], closed
            ),
            mask=struct_column.base_mask,
            offset=struct_column.offset,
            null_count=struct_column.null_count,
            children=struct_column.base_children,
        )

    def copy(self, deep=True):
        struct_copy = super().copy(deep=deep)
        return IntervalColumn(
            size=struct_copy.size,
            dtype=IntervalDtype(
                struct_copy.dtype.fields["left"], self.dtype.closed
            ),
            mask=struct_copy.base_mask,
            offset=struct_copy.offset,
            null_count=struct_copy.null_count,
            children=struct_copy.base_children,
        )

    def as_interval_column(self, dtype):
        if isinstance(dtype, IntervalDtype):
            return IntervalColumn(
                size=self.size,
                dtype=dtype,
                mask=self.mask,
                offset=self.offset,
                null_count=self.null_count,
                children=tuple(
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
