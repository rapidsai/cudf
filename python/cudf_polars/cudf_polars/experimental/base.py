# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
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
    Table data source information.

    Notes
    -----
    This class should be sub-classed for specific
    data source types (e.g. Parquet, DataFrame, etc.).
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


class ColumnSourceInfo:
    """
    Source column information.

    Parameters
    ----------
    table_source_info
        Table data source information.
    column_name
        Column name in the data source.

    Notes
    -----
    This is a thin wrapper around DataSourceInfo that provides
    direct access to column-specific information.
    """

    __slots__ = ("column_name", "table_source_info")
    table_source_info: DataSourceInfo
    column_name: str

    def __init__(self, table_source_info: DataSourceInfo, column_name: str) -> None:
        self.table_source_info = table_source_info
        self.column_name = column_name

    @property
    def row_count(self) -> ColumnStat[int]:
        """Data source row-count estimate."""
        return self.table_source_info.row_count

    @property
    def unique_stats(self) -> UniqueStats:
        """Return unique-value statistics for a column."""
        return self.table_source_info.unique_stats(self.column_name)

    @property
    def storage_size(self) -> ColumnStat[int]:
        """Return the average column size for a single file."""
        return self.table_source_info.storage_size(self.column_name)

    def add_unique_stats_column(self, column: str | None = None) -> None:
        """Add a column needing unique-value information."""
        return self.table_source_info.add_unique_stats_column(
            column or self.column_name
        )


class ColumnStats:
    """
    Column statistics.

    Parameters
    ----------
    name
        Column name.
    children
        Child ColumnStats objects.
    source_info
        Column source information.
    unique_stats
        Unique-value statistics.
    """

    __slots__ = ("children", "name", "source_info", "unique_stats")

    name: str
    children: tuple[ColumnStats, ...]
    source_info: ColumnSourceInfo
    unique_stats: UniqueStats

    def __init__(
        self,
        name: str,
        *,
        children: tuple[ColumnStats, ...] = (),
        source_info: ColumnSourceInfo | None = None,
        unique_stats: UniqueStats | None = None,
    ) -> None:
        self.name = name
        self.children = children
        self.source_info = source_info or ColumnSourceInfo(DataSourceInfo(), name)
        self.unique_stats = unique_stats or UniqueStats()

    def new_parent(
        self,
        *,
        name: str | None = None,
    ) -> ColumnStats:
        """
        Initialize a new parent ColumnStats object.

        Parameters
        ----------
        name
            The new column name.

        Returns
        -------
        A new ColumnStats object.

        Notes
        -----
        This API preserves the original DataSourceInfo reference.
        """
        return ColumnStats(
            name=name or self.name,
            children=(self,),
            # Want to reference the same DataSourceInfo
            source_info=self.source_info,
            # Want fresh UniqueStats so we can mutate in place
            unique_stats=UniqueStats(),
        )


class JoinKey:
    """Join-key information."""

    column_stats: tuple[ColumnStats, ...]
    unique_count_estimate: int | None
    """Estimated unique-value count from join heuristics."""

    def __init__(self, *column_stats: ColumnStats) -> None:
        self.column_stats = column_stats
        self.unique_count_estimate = None

    @cached_property
    def source_row_count(self) -> int | None:
        """Return the estimated number of rows in the join key."""
        return min(
            (
                cs.source_info.row_count.value
                for cs in self.column_stats
                if cs.source_info.row_count.value is not None
            ),
            default=None,
        )

    def names(self) -> list[str]:
        """Return the names of the columns in the join key."""
        return sorted(cs.name for cs in self.column_stats)


class StatsCollector:
    """Column statistics collector."""

    __slots__ = ("column_stats", "join_keys", "joins", "row_count")

    def __init__(self) -> None:
        self.row_count: dict[IR, ColumnStat[int]] = {}
        self.column_stats: dict[IR, dict[str, ColumnStats]] = {}
        self.join_keys: MutableMapping[JoinKey, set[JoinKey]] = defaultdict(
            set[JoinKey]
        )
        # self.join_cols: MutableMapping[ColumnStats, set[ColumnStats]] = defaultdict(
        #     set[ColumnStats]
        # )
        self.joins: dict[IR, list[JoinKey]] = {}
