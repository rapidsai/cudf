# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf.core.accessors.base_accessor import BaseAccessor
from cudf.core.dtypes import StructDtype
from cudf.utils.dtypes import get_dtype_of_same_kind, is_dtype_obj_struct

if TYPE_CHECKING:
    from cudf.core.column.struct import StructColumn
    from cudf.core.dataframe import DataFrame
    from cudf.core.index import Index
    from cudf.core.series import Series


class StructMethods(BaseAccessor):
    """
    Struct methods for Series
    """

    _column: StructColumn

    def __init__(self, parent: Series | Index):
        if not is_dtype_obj_struct(parent.dtype):
            raise AttributeError(
                "Can only use .struct accessor with a 'struct' dtype"
            )
        super().__init__(parent=parent)

    def field(self, key) -> Series | Index:
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
        struct_dtype_fields = StructDtype.from_struct_dtype(
            self._column.dtype
        ).fields
        field_keys = list(struct_dtype_fields.keys())
        if key in struct_dtype_fields:
            pos = field_keys.index(key)
            return self._return_or_inplace(
                self._column.children[pos]._with_type_metadata(
                    get_dtype_of_same_kind(
                        self._column.dtype, struct_dtype_fields[key]
                    )
                )
            )
        elif isinstance(key, int):
            try:
                return self._return_or_inplace(self._column.children[key])
            except IndexError as err:
                raise IndexError(f"Index {key} out of range") from err
        else:
            raise KeyError(
                f"Field '{key}' is not found in the set of existing keys."
            )

    def explode(self) -> DataFrame:
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
        from cudf.core.column_accessor import ColumnAccessor
        from cudf.core.dataframe import DataFrame

        data = {
            name: col.copy(deep=True)
            for name, col in zip(
                self._column.dtype.fields,  # type: ignore[arg-type]
                self._column.children,
                strict=True,
            )
        }
        rangeindex = len(data) == 0
        return DataFrame._from_data(
            ColumnAccessor(data, rangeindex=rangeindex)
        )
