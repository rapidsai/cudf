# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping, Sequence

    from typing_extensions import Self

    import pylibcudf as plc

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.nodebase import Node


@dataclasses.dataclass(frozen=True)
class ColumnStats:
    """Estimated column statistics."""

    dtype: plc.DataType
    """Column data type."""
    unique_count: int
    """Estimated unique count for this column."""
    element_size: int
    """Estimated byte size for each element of this column."""
    file_size: int | None
    """Estimated file size for this column (Optional)."""


@dataclasses.dataclass(frozen=True)
class TableStats:
    """Estimated table statistics."""

    column_stats: dict[str, ColumnStats]
    """Estimated column statistics."""
    num_rows: int
    """Estimated row count."""

    @classmethod
    def merge(
        cls,
        tables: Sequence[TableStats],
        num_rows: int | None = None,
    ) -> Self:
        """
        Merge multiple TableStats objects.

        Parameters
        ----------
        tables
            Sequence of TableStats objects to combine into
            a single TableStats object. Each element
            of ``tables`` will overwrite the ColumnStats
            contributions from previous elements if the
            column names match.
        num_rows
            The estimated row-count for the new TableStats
            object. If nothing is specified, ``num_rows``
            will be set to the maximum found in ``tables``.
        """
        max_num_rows = 0
        column_stats: dict[str, ColumnStats] = {}
        for table_stats in tables:
            column_stats.update(table_stats.column_stats)
            max_num_rows = max(max_num_rows, table_stats.num_rows)
        return cls(column_stats, num_rows or max_num_rows)


class PartitionInfo:
    """Partitioning information."""

    __slots__ = ("count", "partitioned_on", "table_stats")
    count: int
    """Partition count."""
    partitioned_on: tuple[NamedExpr, ...]
    """Columns the data is hash-partitioned on."""
    table_stats: TableStats | None
    """Table statistics (Optional)."""

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
        preserve_partitioned_on: bool = False,
        table_stats: TableStats | None = None,
    ) -> Self:
        """
        Create a new PartitionInfo object.

        Parameters
        ----------
        ir
            The corresponding IR node.
        partition_info
            A mapping from unique IR nodes to the associated
            PartitionInfo object.
        count
            The partition count. By default, the partition
            count will be set to the maximum partition count
            of ``ir.children``. If ``ir`` has no children,
            the default partition count is ``1``.
        partitioned_on
            Columns the data is hash-partitioned on. This will be
            copied from a child of ``ir`` if ``preserve_partitioned_on``
            is set to ``True``.
        preserve_partitioned_on
            Whether to copy ``partitioned_on`` from a child of
            ``ir``. This argument is ignored if ``partitioned_on``
            is not ``None``, or if there are multiple children
            with inconsistent ``partitioned_on`` attributes.
        table_stats
            Table statistics for ``ir``. By default, these statistics
            are copied from ``ir.children``. Copied statistics will
            include all column statistics, and ``num_rows`` will
            be set to the maximum child row-count estimate.

        Returns
        -------
        The new PartitionInfo object, and an updated mapping
        from unique IR nodes to associated PartitionInfo objects.

        Notes
        -----
        This function should be used in lieu of ``PartitionInfo()``
        unless ``ir`` corresponds to a leaf node. This will ensure
        that table statistics are propagated through the IR graph.
        """
        children = ir.children
        count = count or (
            max(partition_info[child].count for child in children) if children else 1
        )
        if preserve_partitioned_on:
            if partitioned_on is not None:  # pragma: no cover
                raise ValueError(
                    "Cannot specify both preserve_partitioned_on and partitioned_on"
                )
            # Inherit partitioned_on
            partitionining = {
                partition_info[child].partitioned_on
                for child in children
                if partition_info[child].partitioned_on
            }
            partitioned_on = partitionining.pop() if len(partitionining) == 1 else ()

        if table_stats is None:
            # Inherit table statistics
            child_table_stats: list[TableStats] = []
            for child in children:
                stats = partition_info[child].table_stats
                if stats is not None:
                    child_table_stats.append(stats)
            if child_table_stats:
                table_stats = TableStats.merge(child_table_stats)

        return cls(count, partitioned_on or (), table_stats)


def get_key_name(node: Node) -> str:
    """Generate the key name for a Node."""
    return f"{type(node).__name__.lower()}-{hash(node)}"
