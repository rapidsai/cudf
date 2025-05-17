# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping

    from typing_extensions import Self

    import pylibcudf as plc

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.nodebase import Node
    from cudf_polars.typing import Schema


@dataclasses.dataclass(frozen=True)
class ColumnStats:
    """Known or estimated column statistics."""

    dtype: plc.DataType
    """Column data type."""
    cardinality: float
    """Known or estimated column cardinality."""
    file_size: int | None
    """Estimated file size for this column."""
    estimated: bool
    """Whether statistics are estimated."""


@dataclasses.dataclass(frozen=True)
class TableStats:
    """Known or estimated table statistics."""

    column_stats: dict[str, ColumnStats]
    """Known or estimated column statistics."""
    num_rows: int | None
    """Known or estimated row count."""
    estimated: bool
    """Whether the row count is estimated."""

    @classmethod
    def merge(cls, schema: Schema, *tables: TableStats) -> Self:
        """Merge multiple TableStats objects."""
        num_rows = 0
        estimated = False
        column_stats: dict[str, ColumnStats] = {}
        for table_stats in tables:
            column_stats.update(
                {k: v for k, v in table_stats.column_stats.items() if k in schema}
            )
            if isinstance(table_stats.num_rows, int):
                num_rows = max(num_rows, table_stats.num_rows)
            estimated = estimated or table_stats.estimated
        return cls(column_stats, num_rows or None, estimated)


class PartitionInfo:
    """Partitioning information."""

    __slots__ = ("count", "partitioned_on", "table_stats")
    count: int
    """Partition count."""
    partitioned_on: tuple[NamedExpr, ...]
    """Columns the data is hash-partitioned on."""
    table_stats: TableStats | None
    """Table statistics."""

    def __init__(
        self,
        count: int,
        partitioned_on: tuple[NamedExpr, ...] = (),
        table_stats: TableStats | None = None,
    ):
        self.count = count
        self.partitioned_on = partitioned_on
        self.table_stats = table_stats

    def keys(self, node: Node) -> Iterator[tuple[str, int]]:
        """Return the partitioned keys for a given node."""
        name = get_key_name(node)
        yield from ((name, i) for i in range(self.count))

    @classmethod
    def new(
        cls,
        ir: IR,
        partition_info: MutableMapping[IR, PartitionInfo],
        *,
        count: int | None = None,
        partitioned_on: tuple[NamedExpr, ...] | None = None,
        inherit_partitioned_on: bool = False,
        table_stats: TableStats | None = None,
        inherit_table_stats: bool = True,
    ) -> Self:
        """Create a new PartitionInfo object."""
        children = ir.children
        count = count or (
            max(partition_info[child].count for child in children) if children else 1
        )
        if partitioned_on is None and inherit_partitioned_on:
            # Inherit partitioned_on
            partitionining = {
                partition_info[child].partitioned_on
                for child in children
                if partition_info[child].partitioned_on
            }
            partitioned_on = partitionining.pop() if len(partitionining) == 1 else ()

        if table_stats is None and inherit_table_stats:
            # Inherit table statistics
            child_table_stats: list[TableStats] = []
            for child in children:
                stats = partition_info[child].table_stats
                if stats is not None:
                    child_table_stats.append(stats)
            if child_table_stats:
                table_stats = TableStats.merge(ir.schema, *child_table_stats)

        return cls(count, partitioned_on or (), table_stats)


def get_key_name(node: Node) -> str:
    """Generate the key name for a Node."""
    return f"{type(node).__name__.lower()}-{hash(node)}"
