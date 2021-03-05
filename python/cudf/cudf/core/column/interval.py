# Copyright (c) 2018-2021, NVIDIA CORPORATION.
import pyarrow as pa
import cudf
import pandas as pd
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
        closed="right",
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
        if closed in ["left", "right", "neither", "both"]:
            self._closed = closed
        else:
            raise ValueError("closed value is not valid")

    @property
    def closed(self):
        return self._closed

    @classmethod
    def from_arrow(self, data):
        new_col = super().from_arrow(data.storage)
        size = len(data)
        dtype = IntervalDtype.from_arrow(data.type)
        mask = data.buffers()[0]
        if mask is not None:
            mask = cudf.utils.utils.pa_mask_buffer_to_mask(mask, len(data))

        offset = data.offset
        null_count = data.null_count
        children = new_col.children
        closed = dtype.closed

        return IntervalColumn(
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
            closed=closed,
        )

    def to_arrow(self):
        typ = self.dtype.to_arrow()
        struct_arrow = super().to_arrow()
        if len(struct_arrow) == 0:
            # struct arrow is pa.struct array with null children types
            # we need to make sure its children have non-null type
            struct_arrow = pa.array([], typ.storage_type)
        return pa.ExtensionArray.from_storage(typ, struct_arrow)

    def from_struct_column(self, closed="right"):
        return IntervalColumn(
            size=self.size,
            dtype=IntervalDtype(self.dtype.fields["left"], closed),
            mask=self.base_mask,
            offset=self.offset,
            null_count=self.null_count,
            children=self.base_children,
            closed=closed,
        )

    def copy(self, deep=True):
        closed = self.closed
        struct_copy = super().copy(deep=deep)
        return IntervalColumn(
            size=struct_copy.size,
            dtype=IntervalDtype(struct_copy.dtype.fields["left"], closed),
            mask=struct_copy.base_mask,
            offset=struct_copy.offset,
            null_count=struct_copy.null_count,
            children=struct_copy.base_children,
            closed=closed,
        )

    def as_interval_column(self, dtype, **kwargs):
        if isinstance(dtype, str) and dtype == "interval":
            return self

        if isinstance(dtype, (IntervalDtype, cudf.core.dtypes.IntervalDtype)):
            return IntervalColumn(
                size=self.size,
                dtype=dtype,
                mask=self.mask,
                offset=self.offset,
                null_count=self.null_count,
                children=self.children,
                closed=dtype.closed,
            )

        if isinstance(dtype, (IntervalDtype, pd.IntervalDtype)):
            return self

        if not isinstance(dtype, IntervalDtype):
            raise ValueError("dtype must be IntervalDtype")
