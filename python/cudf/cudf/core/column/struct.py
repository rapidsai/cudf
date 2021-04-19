# Copyright (c) 2020, NVIDIA CORPORATION.
from __future__ import annotations

import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase
from cudf.core.column.methods import ColumnMethodsMixin
from cudf.utils.dtypes import is_struct_dtype


class StructColumn(ColumnBase):
    """
    Column that stores fields of values.

    Every column has n children, where n is
    the number of fields in the Struct Dtype.

    """

    dtype: cudf.core.dtypes.StructDtype

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

    def struct(self, parent=None):
        return StructMethods(self, parent=parent)

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


class StructMethods(ColumnMethodsMixin):
    """
    Struct methods for Series
    """

    def __init__(self, column, parent=None):
        if not is_struct_dtype(column.dtype):
            raise AttributeError(
                "Can only use .struct accessor with a 'struct' dtype"
            )
        super().__init__(column=column, parent=parent)

    def field(self, key):
        """
        Extract children of the specified struct column
        in the Series

        Parameters
        ----------
        key: int or str
            index/position or field name of the respective
            struct column

        Returns
        -------
        Series

        Examples
        --------
        >>> s = cudf.Series([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
        >>> s.struct.field(0)
        0    1
        1    3
        dtype: int64
        >>> s.struct.field('a')
        0    1
        1    3
        dtype: int64
        """
        fields = list(self._column.dtype.fields.keys())
        if key in fields:
            pos = fields.index(key)
            return self._return_or_inplace(self._column.children[pos])
        else:
            return self._return_or_inplace(self._column.children[key])
