# Copyright (c) 2020, NVIDIA CORPORATION.

import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase


class StructColumn(ColumnBase):
    @property
    def base_size(self):
        if not self.base_children:
            return 0
        else:
            return len(self.base_children[0])

    @classmethod
    def from_arrow(self, data):
        size = len(data)
        dtype = cudf.core.dtypes.StructDtype.from_arrow(data.type)

        mask = data.buffers()[0]
        if mask is not None:
            mask = cudf.utils.utils.pa_mask_buffer_to_mask(mask, len(data))

        offset = data.offset
        null_count = data.null_count
        children = tuple(
            cudf.core.column.as_column(data.field(i))
            for i in range(data.type.num_fields)
        )
        return StructColumn(
            data=None,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    def to_arrow(self):
        children = [
            pa.nulls(len(child))
            if len(child) == child.null_count
            else child.to_arrow()
            for child in self.children
        ]

        pa_type = pa.struct(
            {
                field: child.type
                for field, child in zip(self.dtype.fields, children)
            }
        )

        if self.nullable:
            nbuf = self.mask.to_host_array().view("int8")
            nbuf = pa.py_buffer(nbuf)
            buffers = (nbuf,)
        else:
            buffers = (None,)

        return pa.StructArray.from_buffers(
            pa_type, len(self), buffers, children=children
        )

    def copy(self, deep=True):
        result = super().copy(deep=deep)
        if deep:
            result = result._rename_fields(self.dtype.fields.keys())
        return result

    def _rename_fields(self, names):
        """
        Return a StructColumn with the same field values as this StructColumn,
        but with the field names equal to `names`.
        """
        dtype = cudf.core.dtypes.StructDtype(
            {name: col.dtype for name, col in zip(names, self.children)}
        )
        return StructColumn(
            data=None,
            size=self.base_size,
            dtype=dtype,
            mask=self.base_mask,
            offset=self.offset,
            null_count=self.null_count,
            children=self.base_children,
        )
