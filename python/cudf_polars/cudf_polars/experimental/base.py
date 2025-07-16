# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Generic, TypeVar

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


T = TypeVar("T")


@dataclasses.dataclass
class ColumnStat(Generic[T]):
    """
    Generic column-statistic.

    Parameters
    ----------
    value
        Statistics value. Value will be None
        if the statistics is unknown.
    exact
        Whether the statistics is known exactly.
    """

    value: T | None = None
    exact: bool = False


@dataclasses.dataclass
class UniqueStats:
    """
    Unique-value statistics.

    Parameters
    ----------
    count
        Unique-value count.
    fraction
        Unique-value fraction. This corresponds to the total
        number of unique values (count) divided by the total
        number of rows.
    """

    count: ColumnStat[int] = dataclasses.field(default_factory=ColumnStat[int])
    fraction: ColumnStat[float] = dataclasses.field(default_factory=ColumnStat[float])


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
    def row_count(self) -> ColumnStat[int]:
        """Data source row-count estimate."""
        return ColumnStat[int]()  # pragma: no cover

    def unique_stats(self, column: str) -> UniqueStats:
        """Return unique-value statistics for a column."""
        return UniqueStats()  # pragma: no cover

    def storage_size(self, column: str) -> ColumnStat[int]:
        """Return the average column size for a single file."""
        return ColumnStat[int]()

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
    unique_stats
        Unique-value statistics.
    """

    __slots__ = ("name", "source_info", "source_name", "unique_stats")

    name: str
    source_info: DataSourceInfo
    source_name: str
    unique_stats: UniqueStats

    def __init__(
        self,
        name: str,
        *,
        source_info: DataSourceInfo | None = None,
        source_name: str | None = None,
        unique_stats: UniqueStats | None = None,
    ) -> None:
        self.name = name
        self.source_info = source_info or DataSourceInfo()
        self.source_name = source_name or name
        self.unique_stats = unique_stats or UniqueStats()
