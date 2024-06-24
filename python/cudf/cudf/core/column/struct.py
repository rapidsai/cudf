# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase
from cudf.core.column.methods import ColumnMethods
from cudf.core.dtypes import StructDtype
from cudf.core.missing import NA

if TYPE_CHECKING:
    from cudf._typing import Dtype


class StructColumn(ColumnBase):
    """
    Column that stores fields of values.

    Every column has n children, where n is
    the number of fields in the Struct Dtype.

    """

    dtype: StructDtype

    @property
    def base_size(self):
        if self.base_children:
            return len(self.base_children[0])
        else:
            return self.size + self.offset

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
            buffers = (pa.py_buffer(self.mask.memoryview()),)
        else:
            buffers = (None,)

        return pa.StructArray.from_buffers(
            pa_type, len(self), buffers, children=children
        )

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        # We cannot go via Arrow's `to_pandas` because of the following issue:
        # https://issues.apache.org/jira/browse/ARROW-12680
        if arrow_type or nullable:
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        else:
            return pd.Index(self.to_arrow().tolist(), dtype="object")

    @cached_property
    def memory_usage(self):
        n = 0
        if self.nullable:
            n += cudf._lib.null_mask.bitmask_allocation_size_bytes(self.size)

        for child in self.children:
            n += child.memory_usage

        return n

    def element_indexing(self, index: int):
        result = super().element_indexing(index)
        return {
            field: value
            for field, value in zip(self.dtype.fields, result.values())
        }

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            # filling in fields not in dict
            for field in self.dtype.fields:
                value[field] = value.get(field, NA)

            value = cudf.Scalar(value, self.dtype)
        super().__setitem__(key, value)

    def copy(self, deep=True):
        # Since struct columns are immutable, both deep and
        # shallow copies share the underlying device data and mask.
        result = super().copy(deep=False)
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
            size=self.size,
            dtype=dtype,
            mask=self.base_mask,
            offset=self.offset,
            null_count=self.null_count,
            children=self.base_children,
        )

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            "Structs are not yet supported via `__cuda_array_interface__`"
        )

    def _with_type_metadata(self: StructColumn, dtype: Dtype) -> StructColumn:
        from cudf.core.column import IntervalColumn
        from cudf.core.dtypes import IntervalDtype

        # Check IntervalDtype first because it's a subclass of StructDtype
        if isinstance(dtype, IntervalDtype):
            return IntervalColumn.from_struct_column(self, closed=dtype.closed)
        elif isinstance(dtype, StructDtype):
            return StructColumn(
                data=None,
                dtype=dtype,
                children=tuple(
                    self.base_children[i]._with_type_metadata(dtype.fields[f])
                    for i, f in enumerate(dtype.fields.keys())
                ),
                mask=self.base_mask,
                size=self.size,
                offset=self.offset,
                null_count=self.null_count,
            )

        return self


class StructMethods(ColumnMethods):
    """
    Struct methods for Series
    """

    _column: StructColumn

    def __init__(self, parent=None):
        if not isinstance(parent.dtype, StructDtype):
            raise AttributeError(
                "Can only use .struct accessor with a 'struct' dtype"
            )
        super().__init__(parent=parent)

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
            if isinstance(key, int):
                try:
                    return self._return_or_inplace(self._column.children[key])
                except IndexError:
                    raise IndexError(f"Index {key} out of range")
            else:
                raise KeyError(
                    f"Field '{key}' is not found in the set of existing keys."
                )

    def explode(self):
        """
        Return a DataFrame whose columns are the fields of this struct Series.

        Notes
        -----
        Note that a copy of the columns is made.

        Examples
        --------
        >>> s
        0    {'a': 1, 'b': 'x'}
        1    {'a': 2, 'b': 'y'}
        2    {'a': 3, 'b': 'z'}
        3    {'a': 4, 'b': 'a'}
        dtype: struct

        >>> s.struct.explode()
           a  b
        0  1  x
        1  2  y
        2  3  z
        3  4  a
        """
        return cudf.DataFrame._from_data(
            cudf.core.column_accessor.ColumnAccessor(
                {
                    name: col.copy(deep=True)
                    for name, col in zip(
                        self._column.dtype.fields, self._column.children
                    )
                }
            )
        )
