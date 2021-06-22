# Copyright (c) 2018-2021, NVIDIA CORPORATION.
import pandas as pd
import pyarrow as pa

import cudf
from cudf.core.column import StructColumn
from cudf.core.dtypes import IntervalDtype
from cudf.utils.dtypes import is_interval_dtype


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
        first_field_name = list(self.dtype.fields.keys())[0]
        return IntervalColumn(
            size=self.size,
            dtype=IntervalDtype(self.dtype.fields[first_field_name], closed),
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
        if is_interval_dtype(dtype):
            # a user can directly input the string `interval` as the dtype
            # when creating an interval series or interval dataframe
            if dtype == "interval":
                dtype = IntervalDtype(self.dtype.fields["left"], self.closed)
            return IntervalColumn(
                size=self.size,
                dtype=dtype,
                mask=self.mask,
                offset=self.offset,
                null_count=self.null_count,
                children=self.children,
                closed=dtype.closed,
            )
        else:
            raise ValueError("dtype must be IntervalDtype")

    def to_pandas(self, index: pd.Index = None, **kwargs) -> "pd.Series":
        # Note: This does not handle null values in the interval column.
        # However, this exact sequence (calling __from_arrow__ on the output of
        # self.to_arrow) is currently the best known way to convert interval
        # types into pandas (trying to convert the underlying numerical columns
        # directly is problematic), so we're stuck with this for now.
        return pd.Series(
            pd.IntervalDtype().__from_arrow__(self.to_arrow()), index=index
        )
