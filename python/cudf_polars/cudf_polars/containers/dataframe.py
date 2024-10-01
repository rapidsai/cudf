# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A dataframe, with some properties."""

from __future__ import annotations

import itertools
from functools import cached_property
from typing import TYPE_CHECKING

import pyarrow as pa
import pylibcudf as plc

import polars as pl

from cudf_polars.containers import Column
from cudf_polars.utils import dtypes

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Set

    from typing_extensions import Self


__all__: list[str] = ["DataFrame"]


class DataFrame:
    """A representation of a dataframe."""

    column_map: dict[str, Column]
    table: plc.Table

    def __init__(
        self, columns: Iterable[tuple[str, Column]] | Mapping[str, Column]
    ) -> None:
        self.column_map = dict(columns)
        self.table = plc.Table([c.obj for c in self.column_map.values()])

    def copy(self) -> Self:
        """Return a shallow copy of self."""
        return type(self)((name, c.copy()) for name, c in self.column_map.items())

    def to_polars(self) -> pl.DataFrame:
        """Convert to a polars DataFrame."""
        # If the arrow table has empty names, from_arrow produces
        # column_$i. But here we know there is only one such column
        # (by construction) and it should have an empty name.
        # https://github.com/pola-rs/polars/issues/11632
        # To guarantee we produce correct names, we therefore
        # serialise with names we control and rename with that map.
        name_map = {f"column_{i}": name for i, name in enumerate(self.column_map)}
        table: pa.Table = plc.interop.to_arrow(
            self.table,
            [plc.interop.ColumnMetadata(name=name) for name in name_map],
        )
        df: pl.DataFrame = pl.from_arrow(table)
        return df.rename(name_map).with_columns(
            *(
                pl.col(name).set_sorted(
                    descending=column.order == plc.types.Order.DESCENDING
                )
                if column.is_sorted
                else pl.col(name)
                for name, column in self.column_map.items()
            )
        )

    @cached_property
    def column_names_set(self) -> frozenset[str]:
        """Return the column names as a set."""
        return frozenset(self.column_map)

    @cached_property
    def column_names(self) -> list[str]:
        """Return a list of the column names."""
        return list(self.column_map)

    @cached_property
    def columns(self) -> list[Column]:
        """Return a list of the columns."""
        return list(self.column_map.values())

    @cached_property
    def num_columns(self) -> int:
        """Number of columns."""
        return len(self.column_map)

    @cached_property
    def num_rows(self) -> int:
        """Number of rows."""
        return 0 if len(self.column_map) == 0 else self.table.num_rows()

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> Self:
        """
        Create from a polars dataframe.

        Parameters
        ----------
        df
            Polars dataframe to convert

        Returns
        -------
        New dataframe representing the input.
        """
        table = df.to_arrow()
        schema = table.schema
        for i, field in enumerate(schema):
            schema = schema.set(
                i, pa.field(field.name, dtypes.downcast_arrow_lists(field.type))
            )
        # No-op if the schema is unchanged.
        d_table = plc.interop.from_arrow(table.cast(schema))
        return cls(
            (h_col.name, Column(column).copy_metadata(h_col))
            for column, h_col in zip(d_table.columns(), df.iter_columns(), strict=True)
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
        return cls(zip(names, map(Column, table.columns()), strict=True))

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
        self.column_map = {
            name: column.sorted_like(other) if name in subset else column
            for (name, column), other in zip(
                self.column_map.items(), like.column_map.values(), strict=True
            )
        }
        return self

    def with_columns(self, columns: Iterable[tuple[str, Column]]) -> Self:
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
        return type(self)(itertools.chain(self.column_map.items(), columns))

    def discard_columns(self, names: Set[str]) -> Self:
        """Drop columns by name."""
        return type(self)(
            (name, column)
            for name, column in self.column_map.items()
            if name not in names
        )

    def select(self, names: Sequence[str]) -> Self:
        """Select columns by name returning DataFrame."""
        want = set(names)
        if not want.issubset(self.column_names_set):
            raise ValueError("Can't select missing names")
        return type(self)((name, self.column_map[name]) for name in names)

    def replace_columns(self, *columns: tuple[str, Column]) -> Self:
        """Return a new dataframe with columns replaced by name."""
        new = dict(columns)
        if not set(new).issubset(self.column_names_set):
            raise ValueError("Cannot replace with non-existing names")
        return type(self)(self.column_map | new)

    def rename_columns(self, mapping: Mapping[str, str]) -> Self:
        """Rename some columns."""
        return type(self)(
            (mapping.get(name, name), column)
            for name, column in self.column_map.items()
        )

    def select_columns(self, names: Set[str]) -> list[Column]:
        """Select columns by name."""
        return [c for name, c in self.column_map.items() if name in names]

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
