# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.nodebase import Node


class PartitionInfo:
    """Partitioning information."""

    __slots__ = ("count", "partitioned_on")
    count: int
    """Partition count."""
    partitioned_on: tuple[NamedExpr, ...]
    """Columns the data is hash-partitioned on."""

    def __init__(
        self,
        count: int,
        partitioned_on: tuple[NamedExpr, ...] = (),
    ):
        self.count = count
        self.partitioned_on = partitioned_on

    def keys(self, node: Node) -> Iterator[tuple[str, int]]:
        """Return the partitioned keys for a given node."""
        name = get_key_name(node)
        yield from ((name, i) for i in range(self.count))

    def __rich_repr__(self) -> Generator[Any, None, None]:
        """Formatting for rich.pretty.pprint."""
        yield "count", self.count
        yield "partitioned_on", self.partitioned_on


def get_key_name(node: Node) -> str:
    """Generate the key name for a Node."""
    return f"{type(node).__name__.lower()}-{hash(node)}"


@runtime_checkable
class ColumnStat(Protocol):
    """Generic column-statistic protocol."""

    value: Any
    exact: bool


class IntColumnStat(ColumnStat):
    """Integer column statistic."""

    __slots__ = ("exact", "value")

    def __init__(
        self,
        *,
        value: int | None = None,
        exact: bool = False,
    ):
        self.value = value
        self.exact = exact


class FloatColumnStat(ColumnStat):
    """Float column statistic."""

    __slots__ = ("exact", "value")

    def __init__(
        self,
        *,
        value: float | None = None,
        exact: bool = False,
    ):
        self.value = value
        self.exact = exact


class RowCount(IntColumnStat):
    """
    Row-count statistic.

    Parameters
    ----------
    value
        Row-count value.
    exact
        Whether row-count is known exactly.
    """


class UniqueCount(IntColumnStat):
    """
    Unique-value count statistic.

    Parameters
    ----------
    value
        Unique-value count.
    exact
        Whether unique-value count is known exactly.
    """


class UniqueFraction(FloatColumnStat):
    """
    Unique-value fraction statistic.

    Parameters
    ----------
    value
        Unique-value fraction.
    exact
        Whether unique-value fraction is known exactly.
    """


class StorageSize(IntColumnStat):
    """
    Average storage-size information.

    Parameters
    ----------
    value
        Average file size.
    exact
        Whether the storage size is known exactly.
    """


class DataSourceInfo:
    """
    Datasource information.

    Notes
    -----
    This class should be sub-classed for specific
    datasource types (e.g. Parquet, DataFrame, etc.).
    The required properties/methods enable lazy
    sampling of the underlying datasource.
    """

    @property
    def row_count(self) -> RowCount:
        """Data source row-count estimate."""
        return RowCount()  # pragma: no cover

    def unique_count(self, column: str) -> UniqueCount:
        """Return unique-value count estimate."""
        return UniqueCount()  # pragma: no cover

    def unique_fraction(self, column: str) -> UniqueFraction:
        """Return unique-value fraction estimate."""
        return UniqueFraction()  # pragma: no cover

    def storage_size(self, column: str) -> StorageSize:
        """Return the average column size for a single file."""
        return StorageSize()

    def add_unique_stats_column(self, column: str) -> None:
        """Add a column needing unique-value information."""


class ColumnStats:
    """
    Column statistics.

    Parameters
    ----------
    name
        Column name.
    source
        Datasource information.
    source_name
        Source-column name.
    unique_count
        Unique-value count estimate.
    """

    __slots__ = ("name", "source", "source_name", "unique_count")

    name: str
    source: DataSourceInfo
    source_name: str
    unique_count: UniqueCount

    def __init__(
        self,
        name: str,
        *,
        source: DataSourceInfo | None = None,
        source_name: str | None = None,
        unique_count: UniqueCount | None = None,
    ) -> None:
        self.name = name
        self.source = source or DataSourceInfo()
        self.source_name = source_name or name
        self.unique_count = unique_count or UniqueCount()
