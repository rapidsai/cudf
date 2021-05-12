from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import cudf

if TYPE_CHECKING:
    from cudf.core.column_accessor import ColumnAccessor
    from cudf.internals.arrays import Array


class ArrayAccessor:
    data: List[Array]
    names: List[Any]

    _column_accessor: ColumnAccessor

    def __init__(self, names=None, data=None):
        self.names = names if names is not None else []
        self.data = data if data is not None else []
        assert len(self.data) == len(self.names)

    def to_column_accessor(self):
        return cudf.core.column_accessor.ColumnAccessor(
            {name: data._column for name, data in zip(self.names, self.data)}
        )

    @classmethod
    def from_column_accessor(cls, ca):
        from cudf.internals.arrays import array

        return cls(list(ca.keys()), [array(col) for col in ca.values()])
