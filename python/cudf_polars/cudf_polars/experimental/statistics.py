# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column statistics."""

from __future__ import annotations

import itertools

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    IR,
    ConditionalJoin,
    DataFrameScan,
    GroupBy,
    HConcat,
    HStack,
    Join,
    Scan,
    Select,
    Union,
)
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import (
    ColumnSourceStats,
    ColumnStats,
    StatsCollector,
)
from cudf_polars.experimental.dispatch import add_source_stats
from cudf_polars.experimental.io import _sample_pq_statistics


def collect_source_statistics(root: IR) -> StatsCollector:
    """Collect basic source statistics."""
    stats: StatsCollector = StatsCollector()
    for node in list(traversal([root]))[::-1]:
        add_source_stats(node, stats)
    return stats


@add_source_stats.register(IR)
def _(ir: IR, stats: StatsCollector) -> None:
    # Default `add_source_stats` implementation.
    if len(ir.children) == 1:
        (child,) = ir.children
        child_column_statistics = stats.column_statistics.get(child, {})
        stats.column_statistics[ir] = {
            name: child_column_statistics.get(name, ColumnStats(name=name))
            for name in ir.schema
        }
    else:  # pragma: no cover
        # Multi-child nodes require custom logic
        raise NotImplementedError(
            f"No add_source_stats dispatch registered for {type(ir)}."
        )


@add_source_stats.register(Scan)
def _(ir: Scan, stats: StatsCollector) -> None:
    if ir.typ == "parquet":
        stats.column_statistics[ir] = {
            name: ColumnStats(
                name=name,
                source_stats=css,
            )
            for name, css in _sample_pq_statistics(ir).items()
        }
        if (
            stats.column_statistics[ir]
            and (
                (
                    source_stats := next(
                        iter(stats.column_statistics[ir].values())
                    ).source_stats
                )
                is not None
            )
            and source_stats.cardinality
        ):
            stats.cardinality[ir] = source_stats.cardinality
    else:
        stats.column_statistics[ir] = {
            name: ColumnStats(
                name=name,
                source_stats=ColumnSourceStats(),
            )
            for name in ir.schema
        }


@add_source_stats.register(DataFrameScan)
def _(ir: DataFrameScan, stats: StatsCollector) -> None:
    nrows = ir.df.height()
    stats.column_statistics[ir] = {
        name: ColumnStats(
            name=name,
            source_stats=ColumnSourceStats(
                cardinality=nrows,
                exact=("cardinality",),
            ),
        )
        for name in ir.schema
    }
    stats.cardinality[ir] = nrows


@add_source_stats.register(Join)
def _(ir: Join, stats: StatsCollector) -> None:
    left, right = ir.children
    left_column_statistics = stats.column_statistics.get(left, {})
    right_column_statistics = stats.column_statistics.get(right, {})
    kstats = {
        n.name: left_column_statistics.get(n.name, ColumnStats(name=n.name))
        for n in ir.left_on
    }
    jstats = {
        name: left_column_statistics.get(name, ColumnStats(name=name))
        for name in left.schema
        if name not in kstats
    }
    suffix = ir.options[3]
    jstats |= {
        name if name not in jstats else f"{name}{suffix}": right_column_statistics.get(
            name, ColumnStats(name=name)
        )
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
    child_column_statistics = stats.column_statistics.get(child, {})
    stats.column_statistics[ir] = {
        n.name: child_column_statistics[n.name]
        for n in ir.keys
        if n.name in child_column_statistics
    } | {n.name: ColumnStats(name=n.name) for n in ir.agg_requests}


@add_source_stats.register(HStack)
def _(ir: HStack, stats: StatsCollector) -> None:
    (child,) = ir.children
    child_column_statistics = stats.column_statistics.get(child, {})
    new_cols = {
        n.name: child_column_statistics[n.value.name]
        if isinstance(n.value, expr.Col)
        else ColumnStats(name=n.name)
        for n in ir.columns
    }
    stats.column_statistics[ir] = child_column_statistics | new_cols


@add_source_stats.register(Select)
def _(ir: Select, stats: StatsCollector) -> None:
    (child,) = ir.children
    child_column_statistics = stats.column_statistics.get(child, {})
    stats.column_statistics[ir] = {
        n.name: child_column_statistics[n.value.name]
        if isinstance(n.value, expr.Col)
        else ColumnStats(name=n.name)
        for n in ir.exprs
    }


@add_source_stats.register(HConcat)
def _(ir: HConcat, stats: StatsCollector) -> None:
    stats.column_statistics[ir] = dict(
        itertools.chain.from_iterable(
            stats.column_statistics.get(c, {}).items() for c in ir.children
        )
    )


@add_source_stats.register(Union)
def _(ir: Union, stats: StatsCollector) -> None:
    # TODO: Might be able to preserve source statistics
    stats.column_statistics[ir] = {name: ColumnStats(name=name) for name in ir.schema}
