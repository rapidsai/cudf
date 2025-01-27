# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A dataframe, with some properties."""

from __future__ import annotations

import pickle
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.utils import dtypes

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Set

    from typing_extensions import Self


__all__: list[str] = ["DataFrame"]


# Pacify the type checker. DataFrame init asserts that all the columns
# have a string name, so let's narrow the type.
class NamedColumn(Column):
    name: str


class DataFrame:
    """A representation of a dataframe."""

    column_map: dict[str, Column]
    table: plc.Table
    columns: list[NamedColumn]

    def __init__(self, columns: Iterable[Column]) -> None:
        columns = list(columns)
        if any(c.name is None for c in columns):
            raise ValueError("All columns must have a name")
        self.columns = [cast(NamedColumn, c) for c in columns]
        self.column_map = {c.name: c for c in self.columns}
        self.table = plc.Table([c.obj for c in self.columns])

    def copy(self) -> Self:
        """Return a shallow copy of self."""
        return type(self)(c.copy() for c in self.columns)

    def to_polars(self) -> pl.DataFrame:
        """Convert to a polars DataFrame."""
        # If the arrow table has empty names, from_arrow produces
        # column_$i. But here we know there is only one such column
        # (by construction) and it should have an empty name.
        # https://github.com/pola-rs/polars/issues/11632
        # To guarantee we produce correct names, we therefore
        # serialise with names we control and rename with that map.
        name_map = {f"column_{i}": name for i, name in enumerate(self.column_map)}
        table = plc.interop.to_arrow(
            self.table,
            [plc.interop.ColumnMetadata(name=name) for name in name_map],
        )
        df: pl.DataFrame = pl.from_arrow(table)
        return df.rename(name_map).with_columns(
            pl.col(c.name).set_sorted(descending=c.order == plc.types.Order.DESCENDING)
            if c.is_sorted
            else pl.col(c.name)
            for c in self.columns
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
    def num_columns(self) -> int:
        """Number of columns."""
        return len(self.column_map)

    @cached_property
    def num_rows(self) -> int:
        """Number of rows."""
        return self.table.num_rows() if self.column_map else 0

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
            Column(column).copy_metadata(h_col)
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
        return cls(
            Column(c, name=name) for c, name in zip(table.columns(), names, strict=True)
        )

    @classmethod
    def deserialize(
        cls, header: Mapping[str, Any], frames: tuple[memoryview, plc.gpumemoryview]
    ) -> Self:
        """
        Create a DataFrame from a serialized representation returned by `.serialize()`.

        Parameters
        ----------
        header
            The (unpickled) metadata required to reconstruct the object.
        frames
            Two-tuple of frames (a memoryview and a gpumemoryview).

        Returns
        -------
        DataFrame
            The deserialized DataFrame.
        """
        packed_metadata, packed_gpu_data = frames
        table = plc.contiguous_split.unpack_from_memoryviews(
            packed_metadata, packed_gpu_data
        )
        return cls(
            Column(c, **kw)
            for c, kw in zip(table.columns(), header["columns_kwargs"], strict=True)
        )

    def serialize(
        self,
    ) -> tuple[Mapping[str, Any], tuple[memoryview, plc.gpumemoryview]]:
        """
        Serialize the table into header and frames.

        Follows the Dask serialization scheme with a picklable header (dict) and
        a tuple of frames (in this case a contiguous host and device buffer).

        To enable dask support, dask serializers must be registered

        >>> from cudf_polars.experimental.dask_serialize import register
        >>> register()

        Returns
        -------
        header
            A dict containing any picklable metadata required to reconstruct the object.
        frames
            Two-tuple of frames suitable for passing to `unpack_from_memoryviews`
        """
        packed = plc.contiguous_split.pack(self.table)

        # Keyword arguments for `Column.__init__`.
        columns_kwargs = [
            {
                "is_sorted": col.is_sorted,
                "order": col.order,
                "null_order": col.null_order,
                "name": col.name,
            }
            for col in self.columns
        ]
        header = {
            "columns_kwargs": columns_kwargs,
            # Dask Distributed uses "type-serialized" to dispatch deserialization
            "type-serialized": pickle.dumps(type(self)),
            "frame_count": 2,
        }
        return header, packed.release()

    def sorted_like(
        self, like: DataFrame, /, *, subset: Set[str] | None = None
    ) -> Self:
        """
        Return a shallow copy with sortedness copied from like.

        Parameters
        ----------
        like
            The dataframe to copy from
        subset
            Optional subset of columns from which to copy data.

        Returns
        -------
        Shallow copy of self with metadata set.

        Raises
        ------
        ValueError
            If there is a name mismatch between self and like.
        """
        if like.column_names != self.column_names:
            raise ValueError("Can only copy from identically named frame")
        subset = self.column_names_set if subset is None else subset
        return type(self)(
            c.sorted_like(other) if c.name in subset else c
            for c, other in zip(self.columns, like.columns, strict=True)
        )

    def with_columns(self, columns: Iterable[Column], *, replace_only=False) -> Self:
        """
        Return a new dataframe with extra columns.

        Parameters
        ----------
        columns
            Columns to add
        replace_only
            If true, then only replacements are allowed (matching by name).

        Returns
        -------
        New dataframe

        Notes
        -----
        If column names overlap, newer names replace older ones, and
        appear in the same order as the original frame.
        """
        new = {c.name: c for c in columns}
        if replace_only and not self.column_names_set.issuperset(new.keys()):
            raise ValueError("Cannot replace with non-existing names")
        return type(self)((self.column_map | new).values())

    def discard_columns(self, names: Set[str]) -> Self:
        """Drop columns by name."""
        return type(self)(column for column in self.columns if column.name not in names)

    def select(self, names: Sequence[str]) -> Self:
        """Select columns by name returning DataFrame."""
        try:
            return type(self)(self.column_map[name] for name in names)
        except KeyError as e:
            raise ValueError("Can't select missing names") from e

    def rename_columns(self, mapping: Mapping[str, str]) -> Self:
        """Rename some columns."""
        return type(self)(c.rename(mapping.get(c.name, c.name)) for c in self.columns)

    def select_columns(self, names: Set[str]) -> list[Column]:
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
