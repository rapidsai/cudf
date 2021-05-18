from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import cudf

if TYPE_CHECKING:
    from cudf.internals.arrays import Array


class ArrayAccessor:
    arrays: List[Array]
    names: List[Any]

    def __init__(self, names=None, data=None):
        self.names = list(names) if names is not None else []
        self.arrays = list(data) if data is not None else []
        assert len(self.arrays) == len(self.names)

    def to_column_accessor(self):
        return cudf.core.column_accessor.ColumnAccessor(
            {name: data._column for name, data in zip(self.names, self.arrays)}
        )

    @classmethod
    def from_column_accessor(cls, ca):
        from cudf.internals.arrays import asarray

        return cls(ca.keys(), [asarray(col) for col in ca.values()])

    def __len__(self):
        return len(self.arrays)
