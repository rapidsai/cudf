# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column statistics."""

from __future__ import annotations

import itertools
from functools import singledispatch

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    IR,
    ConditionalJoin,
    DataFrameScan,
    GroupBy,
    HConcat,
    HStack,
    Join,
    Select,
    Union,
)
from cudf_polars.dsl.traversal import post_traversal


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


@singledispatch
def add_source_stats(ir: IR, stats: StatsCollector) -> None:
    """
    Add basic source statistics for an IR node.

    Parameters
    ----------
    ir
        The IR node to collect source statistics for.
    stats
        The `StatsCollector` object to update with new
        source statistics.
    """


def collect_source_statistics(root: IR) -> StatsCollector:
    """Collect basic source statistics."""
    stats: StatsCollector = StatsCollector()
    for node in post_traversal([root]):
        add_source_stats(node, stats)
    return stats


@add_source_stats.register(IR)
def _(ir: IR, stats: StatsCollector) -> None:
    # Default `add_source_stats` implementation.
    if len(ir.children) == 1:
        (child,) = ir.children
        stats.column_statistics[ir] = {
            name: stats.column_statistics[child].get(name, ColumnStats(name=name))
            for name in ir.schema
        }
    else:  # pragma: no cover
        # Multi-child nodes require custom logic
        raise NotImplementedError(
            f"No add_source_stats dispatch registered for {type(ir)}."
        )


@add_source_stats.register(DataFrameScan)
def _(ir: DataFrameScan, stats: StatsCollector) -> None:
    nrows = ir.df.height()
    table_source = TableSourceStats(cardinality=nrows)
    stats.column_statistics[ir] = {
        name: ColumnStats(
            name=name,
            source_stats=ColumnSourceStats(table_source),
        )
        for name in ir.schema
    }
    stats.cardinality[ir] = nrows


@add_source_stats.register(Join)
def _(ir: Join, stats: StatsCollector) -> None:
    left, right = ir.children
    kstats = {
        n.name: stats.column_statistics[left].get(n.name, ColumnStats(name=n.name))
        for n in ir.left_on
    }
    jstats = {
        name: stats.column_statistics[left].get(name, ColumnStats(name=name))
        for name in left.schema
        if name not in kstats
    }
    suffix = ir.options[3]
    jstats |= {
        name if name not in jstats else f"{name}{suffix}": stats.column_statistics[
            right
        ].get(name, ColumnStats(name=name))
        for name in right.schema
        if name not in kstats
    }
    stats.column_statistics[ir] = kstats | jstats


@add_source_stats.register(ConditionalJoin)
def _(ir: Join, stats: StatsCollector) -> None:
    # TODO: Fix this.
    stats.column_statistics[ir] = {name: ColumnStats(name=name) for name in ir.schema}


@add_source_stats.register(GroupBy)
def _(ir: GroupBy, stats: StatsCollector) -> None:
    (child,) = ir.children
    stats.column_statistics[ir] = {
        n.name: stats.column_statistics[child][n.name]
        for n in ir.keys
        if n.name in stats.column_statistics[child]
    } | {n.name: ColumnStats(name=n.name) for n in ir.agg_requests}


@add_source_stats.register(HStack)
def _(ir: HStack, stats: StatsCollector) -> None:
    (child,) = ir.children
    new_cols = {
        n.name: stats.column_statistics[child][n.value.name]
        if isinstance(n.value, expr.Col)
        else ColumnStats(name=n.name)
        for n in ir.columns
    }
    stats.column_statistics[ir] = stats.column_statistics[child] | new_cols


@add_source_stats.register(Select)
def _(ir: Select, stats: StatsCollector) -> None:
    (child,) = ir.children
    stats.column_statistics[ir] = {
        n.name: stats.column_statistics[child][n.value.name]
        if isinstance(n.value, expr.Col)
        else ColumnStats(name=n.name)
        for n in ir.exprs
    }


@add_source_stats.register(HConcat)
def _(ir: HConcat, stats: StatsCollector) -> None:
    stats.column_statistics[ir] = dict(
        itertools.chain.from_iterable(
            stats.column_statistics[c].items() for c in ir.children
        )
    )


@add_source_stats.register(Union)
def _(ir: Union, stats: StatsCollector) -> None:
    # TODO: Might be able to preserve source statistics
    stats.column_statistics[ir] = {name: ColumnStats(name=name) for name in ir.schema}
