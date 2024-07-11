# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A dataframe, with some properties."""

from __future__ import annotations

import itertools
from functools import cached_property
from typing import TYPE_CHECKING, cast

import polars as pl

import cudf._lib.pylibcudf as plc

from cudf_polars.containers.column import NamedColumn

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, Set

    import pyarrow as pa
    from typing_extensions import Self

    import cudf

    from cudf_polars.containers import Column


__all__: list[str] = ["DataFrame"]


class DataFrame:
    """A representation of a dataframe."""

    columns: list[NamedColumn]
    table: plc.Table

    def __init__(self, columns: Sequence[NamedColumn]) -> None:
        self.columns = list(columns)
        self._column_map = {c.name: c for c in self.columns}
        self.table = plc.Table([c.obj for c in columns])

    def copy(self) -> Self:
        """Return a shallow copy of self."""
        return type(self)([c.copy() for c in self.columns])

    def to_polars(self) -> pl.DataFrame:
        """Convert to a polars DataFrame."""
        table: pa.Table = plc.interop.to_arrow(
            self.table,
            [plc.interop.ColumnMetadata(name=c.name) for c in self.columns],
        )
        return cast(pl.DataFrame, pl.from_arrow(table)).with_columns(
            *(
                pl.col(c.name).set_sorted(
                    descending=c.order == plc.types.Order.DESCENDING
                )
                if c.is_sorted
                else pl.col(c.name)
                for c in self.columns
            )
        )

    @cached_property
    def column_names_set(self) -> frozenset[str]:
        """Return the column names as a set."""
        return frozenset(c.name for c in self.columns)

    @cached_property
    def column_names(self) -> list[str]:
        """Return a list of the column names."""
        return [c.name for c in self.columns]

    @cached_property
    def num_columns(self) -> int:
        """Number of columns."""
        return len(self.columns)

    @cached_property
    def num_rows(self) -> int:
        """Number of rows."""
        return 0 if len(self.columns) == 0 else self.table.num_rows()

    @classmethod
    def from_cudf(cls, df: cudf.DataFrame) -> Self:
        """Create from a cudf dataframe."""
        return cls(
            [
                NamedColumn(c.to_pylibcudf(mode="read"), name)
                for name, c in df._data.items()
            ]
        )

    @classmethod
    def from_table(cls, table: plc.Table, names: Sequence[str]) -> Self:
        """
        Create from a pylibcudf table.

        Parameters
        ----------
        table
            Pylibcudf table to obtain columns from
        names
            Names for the columns

        Returns
        -------
        New dataframe sharing data with the input table.

        Raises
        ------
        ValueError
            If the number of provided names does not match the
            number of columns in the table.
        """
        if table.num_columns() != len(names):
            raise ValueError("Mismatching name and table length.")
        return cls(
            # TODO: strict=True when we drop py39
            [NamedColumn(c, name) for c, name in zip(table.columns(), names)]
        )

    def sorted_like(
        self, like: DataFrame, /, *, subset: Set[str] | None = None
    ) -> Self:
        """
        Copy sortedness from a dataframe onto self.

        Parameters
        ----------
        like
            The dataframe to copy from
        subset
            Optional subset of columns from which to copy data.

        Returns
        -------
        Self with metadata set.

        Raises
        ------
        ValueError
            If there is a name mismatch between self and like.
        """
        if like.column_names != self.column_names:
            raise ValueError("Can only copy from identically named frame")
        subset = self.column_names_set if subset is None else subset
        self.columns = [
            c.sorted_like(other) if c.name in subset else c
            # TODO: strict=True when we drop py39
            for c, other in zip(self.columns, like.columns)
        ]
        return self

    def with_columns(self, columns: Sequence[NamedColumn]) -> Self:
        """
        Return a new dataframe with extra columns.

        Parameters
        ----------
        columns
            Columns to add

        Returns
        -------
        New dataframe

        Notes
        -----
        If column names overlap, newer names replace older ones.
        """
        columns = list(
            {c.name: c for c in itertools.chain(self.columns, columns)}.values()
        )
        return type(self)(columns)

    def discard_columns(self, names: Set[str]) -> Self:
        """Drop columns by name."""
        return type(self)([c for c in self.columns if c.name not in names])

    def select(self, names: Sequence[str]) -> Self:
        """Select columns by name returning DataFrame."""
        want = set(names)
        if not want.issubset(self.column_names_set):
            raise ValueError("Can't select missing names")
        return type(self)([self._column_map[name] for name in names])

    def replace_columns(self, *columns: NamedColumn) -> Self:
        """Return a new dataframe with columns replaced by name."""
        new = {c.name: c for c in columns}
        if not set(new).issubset(self.column_names_set):
            raise ValueError("Cannot replace with non-existing names")
        return type(self)([new.get(c.name, c) for c in self.columns])

    def rename_columns(self, mapping: Mapping[str, str]) -> Self:
        """Rename some columns."""
        return type(self)([c.copy(new_name=mapping.get(c.name)) for c in self.columns])

    def select_columns(self, names: Set[str]) -> list[NamedColumn]:
        """Select columns by name."""
        return [c for c in self.columns if c.name in names]

    def filter(self, mask: Column) -> Self:
        """Return a filtered table given a mask."""
        table = plc.stream_compaction.apply_boolean_mask(self.table, mask.obj)
        return type(self).from_table(table, self.column_names).sorted_like(self)

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
        New dataframe (if zlice is not None) otherwise self (if it is)
        """
        if zlice is None:
            return self
        start, length = zlice
        if start < 0:
            start += self.num_rows
        # Polars implementation wraps negative start by num_rows, then
        # adds length to start to get the end, then clamps both to
        # [0, num_rows)
        end = start + length
        start = max(min(start, self.num_rows), 0)
        end = max(min(end, self.num_rows), 0)
        (table,) = plc.copying.slice(self.table, [start, end])
        return type(self).from_table(table, self.column_names).sorted_like(self)
