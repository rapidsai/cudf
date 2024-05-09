# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A dataframe, with some properties."""

from __future__ import annotations

import itertools
from functools import cached_property
from typing import TYPE_CHECKING

import cudf._lib.pylibcudf as plc

from cudf_polars.containers.column import Column
from cudf_polars.containers.scalar import Scalar

if TYPE_CHECKING:
    from typing_extensions import Self

    import cudf


__all__: list[str] = ["DataFrame"]


class DataFrame:
    """A representation of a dataframe."""

    __slots__ = ("columns", "scalars", "names", "scalar_names", "table")
    columns: list[Column]
    scalars: list[Scalar]
    names: dict[str, int]
    scalar_names: frozenset[str]
    table: plc.Table | None

    def __init__(self, columns: list[Column], scalars: list[Scalar]) -> None:
        self.names = dict(zip((c.name for c in columns), itertools.count(0))) | dict(
            zip((s.name for s in columns), itertools.count(0))
        )
        self.scalar_names = frozenset(s.name for s in scalars)
        self.columns = columns
        self.scalars = scalars
        if len(scalars) == 0:
            self.table = plc.Table([c.obj for c in columns])
        else:
            self.table = None

    __iter__ = None

    def __getitem__(self, name: str) -> Column | Scalar:
        """Return column with given name."""
        i = self.names[name]
        if name in self.scalar_names:
            return self.scalars[i]
        else:
            return self.columns[i]

    @cached_property
    def num_rows(self):
        """Number of rows."""
        if self.table is None:
            raise ValueError("Number of rows of frame with scalars makes no sense")
        return self.table.num_rows()

    @classmethod
    def from_cudf(cls, df: cudf.DataFrame) -> Self:
        """Create from a cudf dataframe."""
        return cls(
            [Column(c.to_pylibcudf(mode="read"), name) for name, c in df._data.items()],
            [],
        )

    def with_columns(self, *columns: Column | Scalar) -> Self:
        """
        Return a new dataframe with extra columns.

        Data is shared.
        """
        cols = [c for c in columns if isinstance(c, Column)]
        scalars = [c for c in columns if isinstance(c, Scalar)]
        return type(self)([*self.columns, *cols], [*self.scalars, *scalars])

    def discard_columns(self, names: set[str]) -> Self:
        """Drop columns by name."""
        return type(self)([c for c in self.columns if c not in names], self.scalars)

    def replace_columns(self, *columns: Column) -> Self:
        """Return a new dataframe with columns replaced by name, maintaining order."""
        new = {c.name: c for c in columns}
        if set(new).intersection(self.scalar_names):
            raise ValueError("Cannot replace scalars")
        if not set(new).issubset(self.names):
            raise ValueError("Cannot replace with non-existing names")
        return type(self)([new.get(c.name, c) for c in self.columns], self.scalars)

    def rename_columns(self, mapping: dict[str, str]) -> Self:
        """Rename some columns."""
        new_columns = [
            Column(c, mapping.get(c.name, c.name)).with_metadata(like=c)
            for c in self.columns
        ]
        return type(self)(new_columns, self.scalars)

    def select_columns(self, names: set[str]) -> list[Column]:
        """Select columns by name."""
        return [c for c in self.columns if c.name in names]

    def filter(self, mask: Column) -> Self:
        """Return a filtered table given a mask."""
        table = plc.stream_compaction.apply_boolean_mask(self.table, mask.obj)
        return type(self)(
            [
                Column(new, old.name).with_metadata(like=old)
                for old, new in zip(self.columns, table.columns())
            ],
            [],
        )
