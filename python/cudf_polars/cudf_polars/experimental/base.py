# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

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


class ColumnSourceStats:
    """
    Column source statistics.

    Parameters
    ----------
    cardinality
        Cardinality (row count).
    unique_count
        Unique-value count.
    unique_fraction
        Unique-value fraction.
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
        "cardinality",
        "exact",
        "storage_size_per_file",
        "unique_count",
        "unique_fraction",
    )

    def __init__(
        self,
        *,
        cardinality: int | None = None,
        storage_size_per_file: int | None = None,
        unique_count: int | None = None,
        unique_fraction: float | None = None,
        exact: tuple[str, ...] = (),
    ):
        self.cardinality = cardinality
        self.storage_size_per_file = storage_size_per_file
        self.unique_count = unique_count
        self.unique_fraction = unique_fraction
        self.exact = exact


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
