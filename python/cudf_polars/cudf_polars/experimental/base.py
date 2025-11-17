# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

import dataclasses
import enum
from collections import defaultdict
from enum import IntEnum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.nodebase import Node


class PartitionInfo:
    """Partitioning information."""

    __slots__ = ("count", "io_plan", "partitioned_on")
    count: int
    """Partition count."""
    partitioned_on: tuple[NamedExpr, ...]
    """Columns the data is hash-partitioned on."""
    io_plan: IOPartitionPlan | None
    """IO partitioning plan (Scan nodes only)."""

    def __init__(
        self,
        count: int,
        *,
        partitioned_on: tuple[NamedExpr, ...] = (),
        io_plan: IOPartitionPlan | None = None,
    ):
        self.count = count
        self.partitioned_on = partitioned_on
        self.io_plan = io_plan

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
    Sampled unique-value statistics.

    Parameters
    ----------
    count
        Unique-value count.
    fraction
        Unique-value fraction. This corresponds to the total
        number of unique values (count) divided by the total
        number of rows.

    Notes
    -----
    This class is used to track unique-value column statistics
    that have been sampled from a data source.
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

    _unique_stats_columns: set[str]

    @property
    def row_count(self) -> ColumnStat[int]:  # pragma: no cover
        """Data source row-count estimate."""
        raise NotImplementedError("Sub-class must implement row_count.")

    def unique_stats(
        self,
        column: str,
    ) -> UniqueStats:  # pragma: no cover
        """Return unique-value statistics for a column."""
        raise NotImplementedError("Sub-class must implement unique_stats.")

    def storage_size(self, column: str) -> ColumnStat[int]:
        """Return the average column size for a single file."""
        return ColumnStat[int]()

    @property
    def unique_stats_columns(self) -> set[str]:
        """Return the set of columns needing unique-value information."""
        return self._unique_stats_columns

    def add_unique_stats_column(self, column: str) -> None:
        """Add a column needing unique-value information."""
        self._unique_stats_columns.add(column)


class DataSourcePair(NamedTuple):
    """Pair of table-source and column-name information."""

    table_source: DataSourceInfo
    column_name: str


class ColumnSourceInfo:
    """
    Source column information.

    Parameters
    ----------
    table_source_pairs
        Sequence of DataSourcePair objects.
        Union operations will result in multiple elements.

    Notes
    -----
    This is a thin wrapper around DataSourceInfo that provides
    direct access to column-specific information.
    """

    __slots__ = (
        "implied_unique_count",
        "table_source_pairs",
    )
    table_source_pairs: list[DataSourcePair]
    implied_unique_count: ColumnStat[int]
    """Unique-value count implied by join heuristics."""

    def __init__(self, *table_source_pairs: DataSourcePair) -> None:
        self.table_source_pairs = list(table_source_pairs)
        self.implied_unique_count = ColumnStat[int](None)

    @property
    def is_unique_stats_column(self) -> bool:
        """Return whether this column requires unique-value information."""
        return any(
            pair.column_name in pair.table_source.unique_stats_columns
            for pair in self.table_source_pairs
        )

    @property
    def row_count(self) -> ColumnStat[int]:
        """Data source row-count estimate."""
        return ColumnStat[int](
            # Use sum of table-source row-count estimates.
            value=sum(
                value
                for pair in self.table_source_pairs
                if (value := pair.table_source.row_count.value) is not None
            )
            or None,
            # Row-count may be exact if there is only one table source.
            exact=len(self.table_source_pairs) == 1
            and self.table_source_pairs[0].table_source.row_count.exact,
        )

    def unique_stats(self, *, force: bool = False) -> UniqueStats:
        """
        Return unique-value statistics for a column.

        Parameters
        ----------
        force
            If True, return unique-value statistics even if the column
            wasn't marked as needing unique-value information.
        """
        if (force or self.is_unique_stats_column) and len(self.table_source_pairs) == 1:
            # Single table source.
            # TODO: Handle multiple tables sources if/when necessary.
            # We may never need to do this if the source unique-value
            # statistics are only "used" by the Scan/DataFrameScan nodes.
            table_source, column_name = self.table_source_pairs[0]
            return table_source.unique_stats(column_name)
        else:
            # Avoid sampling unique-stats if this column
            # wasn't marked as "needing" unique-stats.
            return UniqueStats()

    @property
    def storage_size(self) -> ColumnStat[int]:
        """Return the average column size for a single file."""
        # We don't need to handle concatenated statistics for ``storage_size``.
        # Just return the storage size of the first table source.
        if self.table_source_pairs:
            table_source, column_name = self.table_source_pairs[0]
            return table_source.storage_size(column_name)
        else:  # pragma: no cover; We never call this for empty table sources.
            return ColumnStat[int]()

    def add_unique_stats_column(self, column: str | None = None) -> None:
        """Add a column needing unique-value information."""
        # We must call add_unique_stats_column for ALL table sources.
        for table_source, column_name in self.table_source_pairs:
            table_source.add_unique_stats_column(column or column_name)


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
    unique_count
        Unique-value count.
    """

    __slots__ = ("children", "name", "source_info", "unique_count")

    name: str
    children: tuple[ColumnStats, ...]
    source_info: ColumnSourceInfo
    unique_count: ColumnStat[int]

    def __init__(
        self,
        name: str,
        *,
        children: tuple[ColumnStats, ...] = (),
        source_info: ColumnSourceInfo | None = None,
        unique_count: ColumnStat[int] | None = None,
    ) -> None:
        self.name = name
        self.children = children
        self.source_info = source_info or ColumnSourceInfo()
        self.unique_count = unique_count or ColumnStat[int](None)

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
        )


class JoinKey:
    """
    Join-key information.

    Parameters
    ----------
    column_stats
        Column statistics for the join key.

    Notes
    -----
    This class is used to track join-key information.
    It is used to track the columns being joined on
    and the estimated unique-value count for the join key.
    """

    column_stats: tuple[ColumnStats, ...]
    implied_unique_count: int | None
    """Estimated unique-value count from join heuristics."""

    def __init__(self, *column_stats: ColumnStats) -> None:
        self.column_stats = column_stats
        self.implied_unique_count = None

    @cached_property
    def source_row_count(self) -> int | None:
        """
        Return the estimated row-count of the source columns.

        Notes
        -----
        This is the maximum row-count estimate of the source columns.
        """
        return max(
            (
                cs.source_info.row_count.value
                for cs in self.column_stats
                if cs.source_info.row_count.value is not None
            ),
            default=None,
        )


class JoinInfo:
    """
    Join information.

    Notes
    -----
    This class is used to track mappings between joined-on
    columns and joined-on keys (groups of columns). We need
    these mappings to calculate equivalence sets and make
    join-based unique-count and row-count estimates.
    """

    __slots__ = ("column_map", "join_map", "key_map")

    column_map: MutableMapping[ColumnStats, set[ColumnStats]]
    """Mapping between joined columns."""
    key_map: MutableMapping[JoinKey, set[JoinKey]]
    """Mapping between joined keys (groups of columns)."""
    join_map: dict[IR, list[JoinKey]]
    """Mapping between IR nodes and associated join keys."""

    def __init__(self) -> None:
        self.column_map: MutableMapping[ColumnStats, set[ColumnStats]] = defaultdict(
            set[ColumnStats]
        )
        self.key_map: MutableMapping[JoinKey, set[JoinKey]] = defaultdict(set[JoinKey])
        self.join_map: dict[IR, list[JoinKey]] = {}


class StatsCollector:
    """Column statistics collector."""

    __slots__ = ("column_stats", "join_info", "row_count")

    row_count: dict[IR, ColumnStat[int]]
    """Estimated row count for each IR node."""
    column_stats: dict[IR, dict[str, ColumnStats]]
    """Column statistics for each IR node."""
    join_info: JoinInfo
    """Join information."""

    def __init__(self) -> None:
        self.row_count: dict[IR, ColumnStat[int]] = {}
        self.column_stats: dict[IR, dict[str, ColumnStats]] = {}
        self.join_info = JoinInfo()


class IOPartitionFlavor(IntEnum):
    """Flavor of IO partitioning."""

    SINGLE_FILE = enum.auto()  # 1:1 mapping between files and partitions
    SPLIT_FILES = enum.auto()  # Split each file into >1 partition
    FUSED_FILES = enum.auto()  # Fuse multiple files into each partition
    SINGLE_READ = enum.auto()  # One worker/task reads everything


class IOPartitionPlan:
    """
    IO partitioning plan.

    Notes
    -----
    The meaning of `factor` depends on the value of `flavor`:
      - SINGLE_FILE: `factor` must be `1`.
      - SPLIT_FILES: `factor` is the number of partitions per file.
      - FUSED_FILES: `factor` is the number of files per partition.
      - SINGLE_READ: `factor` is the total number of files.
    """

    __slots__ = ("factor", "flavor")
    factor: int
    flavor: IOPartitionFlavor

    def __init__(self, factor: int, flavor: IOPartitionFlavor) -> None:
        if flavor == IOPartitionFlavor.SINGLE_FILE and factor != 1:  # pragma: no cover
            raise ValueError(f"Expected factor == 1 for {flavor}, got: {factor}")
        self.factor = factor
        self.flavor = flavor
