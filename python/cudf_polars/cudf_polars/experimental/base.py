# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

import enum
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

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


class DataSourceInfo(Protocol):
    """
    Table data source information.

    Notes
    -----
    Sub-class for specific data source types (e.g. Parquet, DataFrame).
    """

    @property
    def type(self) -> Literal["parquet", "dataframe"]:
        """The type of the data source. Useful for serialization and deserialization."""

    @property
    def row_count(self) -> int | None:
        """Data source row-count estimate."""

    def column_storage_size(self, column: str) -> int | None:
        """Return the average storage size for a single column in one file."""

    def serialize(self) -> dict[str, Any]:
        """Return JSON-serializable representation of the data source info."""

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DataSourceInfo:
        """Deserialize a DataSourceInfo from a dictionary."""


class StatsCollector:
    """Scan statistics collector."""

    __slots__ = ("scan_stats",)

    scan_stats: dict[IR, DataSourceInfo]
    """DataSourceInfo for each leaf Scan/DataFrameScan node."""

    def __init__(self) -> None:
        self.scan_stats: dict[IR, DataSourceInfo] = {}


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
