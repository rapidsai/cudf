# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A dataframe, with some properties."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import cudf._lib.pylibcudf as plc

from cudf_polars.containers.column import Column

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    import cudf

    from cudf_polars.containers.scalar import Scalar


__all__: list[str] = ["DataFrame"]


class DataFrame:
    """A representation of a dataframe."""

    __slots__ = ("columns", "scalars", "names", "scalar_names", "table")
    columns: list[Column]
    scalars: list[Scalar]
    scalar_names: frozenset[str]
    table: plc.Table | None

    def __init__(self, columns: list[Column], scalars: list[Scalar]) -> None:
        self.scalar_names = frozenset(s.name for s in scalars)
        self.columns = columns
        self.scalars = scalars
        if len(scalars) == 0:
            self.table = plc.Table([c.obj for c in columns])
        else:
            self.table = None

    __iter__ = None

    @cached_property
    def column_names_set(self) -> set[str]:
        """Return the column names as a set."""
        return {c.name for c in self.columns}

    @cached_property
    def column_names(self) -> list[str]:
        """Return a list of the column names."""
        return [c.name for c in self.columns]

    @cached_property
    def num_columns(self):
        """Number of columns."""
        return len(self.columns)

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

    @classmethod
    def from_table(cls, table: plc.Table, names: list[str]) -> Self:
        """Create from a pylibcudf table."""
        if table.num_columns != len(names):
            raise ValueError("Mismatching name and table length.")
        return cls([Column(c, name) for c, name in zip(table.columns(), names)], [])

    def with_sorted(self, *, like: DataFrame, subset: set[str] | None = None) -> Self:
        """Copy sortedness from a dataframe onto self."""
        if like.column_names != self.column_names:
            raise ValueError("Can only copy from identically named frame")
        subset = self.column_names_set if subset is None else subset
        self.columns = [
            c.with_sorted(like=other) if c.name in subset else c
            for c, other in zip(self.columns, like.columns)
        ]
        return self

    def with_columns(self, columns: Sequence[Column]) -> Self:
        """
        Return a new dataframe with extra columns.

        Data is shared.
        """
        return type(self)([*self.columns, *columns], self.scalars)

    def discard_columns(self, names: set[str]) -> Self:
        """Drop columns by name."""
        return type(self)(
            [c for c in self.columns if c.name not in names], self.scalars
        )

    def select(self, names: Sequence[str]) -> Self:
        """Select columns by name returning DataFrame."""
        want = set(names)
        return type(self)([c for c in self.columns if c.name in want], self.scalars)

    def replace_columns(self, *columns: Column) -> Self:
        """Return a new dataframe with columns replaced by name."""
        new = {c.name: c for c in columns}
        if set(new).intersection(self.scalar_names):
            raise ValueError("Cannot replace scalars")
        if not set(new).issubset(self.column_names_set):
            raise ValueError("Cannot replace with non-existing names")
        return type(self)([new.get(c.name, c) for c in self.columns], self.scalars)

    def rename_columns(self, mapping: dict[str, str]) -> Self:
        """Rename some columns."""
        return type(self)(
            [c.rename(mapping.get(c.name, c.name)) for c in self.columns], self.scalars
        )

    def select_columns(self, names: set[str]) -> list[Column]:
        """Select columns by name."""
        return [c for c in self.columns if c.name in names]

    def filter(self, mask: Column) -> Self:
        """Return a filtered table given a mask."""
        table = plc.stream_compaction.apply_boolean_mask(self.table, mask.obj)
        return type(self).from_table(table, self.column_names).with_sorted(like=self)

    def slice(self, zlice: tuple[int, int] | None) -> Self:
        """
        Slice a dataframe.

        Parameters
        ----------
        zlice
            optional, tuple of start and length, negative values of start
            treated as for python indexing. If not provided, returns self.

        Returns
        -------
        New dataframe (if zlice is not None) other self (if it is)
        """
        if zlice is None:
            return self
        start, length = zlice
        if start < 0:
            start += self.num_rows
        # Polars slice takes an arbitrary positive integer and slice
        # to the end of the frame if it is larger.
        end = min(start + length, self.num_rows)
        (table,) = plc.copying.slice(self.table, [start, end])
        return type(self).from_table(table, self.column_names).with_sorted(like=self)
