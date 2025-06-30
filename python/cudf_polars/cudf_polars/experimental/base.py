# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

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


class UniqueSourceStats:
    """
    Unique source statistics.

    Parameters
    ----------
    count
        Unique-value count.
    fraction
        Unique-value fraction.
    """

    __slots__ = ("count", "fraction")

    def __init__(
        self,
        *,
        count: int | None = None,
        fraction: float | None = None,
    ):
        self.count = count
        self.fraction = fraction


class ColumnSourceStats:
    """
    Column source statistics.

    Parameters
    ----------
    cardinality
        Cardinality (row count).
    unique_stats
        Unique-value statistics.
    storage_size_per_file
        Average un-compressed storage size for this
        column in a single file. This value is used to
        calculate the partition count for an IR node.
    exact
        Tuple of attributes that have not been estimated
        by partial sampling, and are known exactly,

    Notes
    -----
    Source statistics are statistics coming from "source"
    nodes like ``Scan` and ``DataFrameScan``.
    """

    __slots__ = (
        "_unique_stats",
        "cardinality",
        "exact",
        "storage_size_per_file",
    )

    def __init__(
        self,
        *,
        cardinality: int | None = None,
        storage_size_per_file: int | None = None,
        exact: tuple[str, ...] = (),
        unique_stats: Callable[..., UniqueSourceStats]
        | UniqueSourceStats
        | None = None,
    ):
        self.cardinality = cardinality
        self.storage_size_per_file = storage_size_per_file
        self.exact = exact
        self._unique_stats = unique_stats

    @property
    def unique_stats(self) -> UniqueSourceStats:
        """Get unique-value statistics."""
        if self._unique_stats is None:
            return UniqueSourceStats()
        if isinstance(self._unique_stats, UniqueSourceStats):
            return self._unique_stats
        elif callable(self._unique_stats):
            return self._unique_stats()
        else:
            raise TypeError()

    @property
    def unique_count(self) -> int | None:
        """Get unique count."""
        return self.unique_stats.count

    @property
    def unique_fraction(self) -> float | None:
        """Get unique fraction."""
        return self.unique_stats.fraction


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

    __slots__ = ("cardinality", "column_stats")

    def __init__(self) -> None:
        self.cardinality: dict[IR, int] = {}
        self.column_stats: dict[IR, dict[str, ColumnStats]] = {}


# class SourceInfo:
#     paths: tuple[str]
#     columns: set[str]
#     key_columns: set[str]

#     def __init__(self, paths: Sequence[str], columns: Sequence[str], key_columns: set[str]):
#         self.paths = tuple(paths)
#         self.columns = set(columns)
#         self.key_columns = set(key_columns)

#     def update(self, columns: Sequence[str], key_columns: Sequence[str]):
#         new_columns = self.columns.union(columns)
#         new_key_columns = self.key_columns.union(key_columns)
#         if new_columns - self.columns or new_key_columns - self.key_columns:
#             return SourceInfo(self.paths, new_columns, new_key_columns)
#         return self
