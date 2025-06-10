# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def get_key_name(node: Node) -> str:
    """Generate the key name for a Node."""
    return f"{type(node).__name__.lower()}-{hash(node)}"


class TableSourceStats:
    """
    Table source statistics.

    Parameters
    ----------
    paths
        Storage path names. If None, the data originated
        from an in-memory source.
    cardinality
        Cardinality (row count) of the data source. This
        value corresponds to the cardinality before any
        filtering or slicing has occurred. If None, the
        cardinality is unknown.
    """

    __slots__ = ("cardinality", "paths")

    def __init__(
        self,
        *,
        paths: tuple[str, ...] = (),
        cardinality: int | None = None,
    ):
        self.paths = paths
        self.cardinality = cardinality


class ColumnSourceStats:
    """
    Column source statistics.

    Parameters
    ----------
    table_source
        Table-source information.
    unique_count
        Unique-count estimate.
    unique_fraction
        Unique-fraction estimate.
    file_size
        Estimated un-compressed storage size for this
        column in a single file. This value is used to
        calculate the partition count for an IR node.
    """

    __slots__ = ("file_size", "table_source", "unique_count", "unique_fraction")

    def __init__(
        self,
        table_source: TableSourceStats,
        *,
        unique_count: int | None = None,
        unique_fraction: float | None = None,
        file_size: int | None = None,
    ):
        self.table_source = table_source
        self.unique_count = unique_count
        self.unique_fraction = unique_fraction
        self.file_size = file_size


class ColumnStats:
    """
    Column statistics.

    Parameters
    ----------
    name
        Column name.
    unique_count
        Unique-count estimate.
    source_stats
        Column-source statistics.
    """

    __slots__ = ("name", "source_stats", "unique_count")

    def __init__(
        self,
        *,
        name: str | None = None,
        unique_count: int | None = None,
        source_stats: ColumnSourceStats | None = None,
    ) -> None:
        self.name = name
        self.unique_count = unique_count
        self.source_stats = source_stats


class StatsCollector:
    """Column statistics collector."""

    __slots__ = ("cardinality", "column_statistics")

    def __init__(self) -> None:
        self.cardinality: dict[IR, int] = {}
        self.column_statistics: dict[IR, dict[str, ColumnStats]] = {}
