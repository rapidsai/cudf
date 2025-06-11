# Copyright (c) 2019-2025, NVIDIA CORPORATION.

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf.core.column.methods import ColumnMethods
from cudf.utils.dtypes import (
    is_dtype_obj_string,
)

if TYPE_CHECKING:
    from cudf._typing import (
        SeriesOrIndex,
    )
    from cudf.core.column.string import StringColumn


class StringMethods(ColumnMethods):
    """
    String methods for Series
    """

    _column: StringColumn

    def __init__(self, parent):
        if not is_dtype_obj_string(parent.dtype):
            raise AttributeError(
                "Can only use .str accessor with a 'string' dtype"
            )
        super().__init__(parent=parent)

    def htoi(self) -> SeriesOrIndex:
        """
        Returns integer value represented by each hex string.
        String is interpreted to have hex (base-16) characters.

        Returns
        -------
        Series/Index of str dtype

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["1234", "ABCDEF", "1A2", "cafe"])
        >>> s.str.htoi()
        0        4660
        1    11259375
        2         418
        3       51966
        dtype: int64
        """
        return self._return_or_inplace(self._column.hex_to_integers())

    hex_to_int = htoi
