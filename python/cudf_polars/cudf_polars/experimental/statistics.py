# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column statistics."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

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
from cudf_polars.experimental.io import _sample_pq_stats

if TYPE_CHECKING:
    from cudf_polars.utils.config import ConfigOptions


def collect_source_stats(root: IR, config_options: ConfigOptions) -> StatsCollector:
    """Collect basic source statistics."""
    stats: StatsCollector = StatsCollector(config_options)
    for node in list(traversal([root]))[::-1]:
        add_source_stats(node, stats)
    return stats


@add_source_stats.register(IR)
def _(ir: IR, stats: StatsCollector) -> None:
    # Default `add_source_stats` implementation.
    if len(ir.children) == 1:
        (child,) = ir.children
        child_column_stats = stats.column_stats.get(child, {})
        stats.column_stats[ir] = {
            name: child_column_stats.get(name, ColumnStats(name=name))
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
        stats.column_stats[ir] = {
            name: ColumnStats(
                name=name,
                source_stats=css,
            )
            for name, css in _sample_pq_stats(ir, stats.config_options).items()
        }
        if (
            stats.column_stats[ir]
            and (
                (
                    source_stats := next(
                        iter(stats.column_stats[ir].values())
                    ).source_stats
                )
                is not None
            )
            and source_stats.cardinality
        ):
            stats.cardinality[ir] = source_stats.cardinality
    else:
        stats.column_stats[ir] = {
            name: ColumnStats(
                name=name,
                source_stats=ColumnSourceStats(),
            )
            for name in ir.schema
        }


@add_source_stats.register(DataFrameScan)
def _(ir: DataFrameScan, stats: StatsCollector) -> None:
    nrows = ir.df.height()
    stats.column_stats[ir] = {
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
    left_column_stats = stats.column_stats.get(left, {})
    right_column_stats = stats.column_stats.get(right, {})
    kstats = {
        n.name: left_column_stats.get(n.name, ColumnStats(name=n.name))
        for n in ir.left_on
    }
    jstats = {
        name: left_column_stats.get(name, ColumnStats(name=name))
        for name in left.schema
        if name not in kstats
    }
    suffix = ir.options[3]
    jstats |= {
        name if name not in jstats else f"{name}{suffix}": right_column_stats.get(
            name, ColumnStats(name=name)
        )
        for name in right.schema
        if name not in kstats
    }
    stats.column_stats[ir] = kstats | jstats


@add_source_stats.register(ConditionalJoin)
def _(ir: Join, stats: StatsCollector) -> None:
    # TODO: Fix this.
    stats.column_stats[ir] = {name: ColumnStats(name=name) for name in ir.schema}


@add_source_stats.register(GroupBy)
def _(ir: GroupBy, stats: StatsCollector) -> None:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    stats.column_stats[ir] = {
        n.name: child_column_stats[n.name]
        for n in ir.keys
        if n.name in child_column_stats
    } | {n.name: ColumnStats(name=n.name) for n in ir.agg_requests}


@add_source_stats.register(HStack)
def _(ir: HStack, stats: StatsCollector) -> None:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    new_cols = {
        n.name: child_column_stats[n.value.name]
        if isinstance(n.value, expr.Col)
        else ColumnStats(name=n.name)
        for n in ir.columns
    }
    stats.column_stats[ir] = child_column_stats | new_cols


@add_source_stats.register(Select)
def _(ir: Select, stats: StatsCollector) -> None:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    stats.column_stats[ir] = {
        n.name: child_column_stats[n.value.name]
        if isinstance(n.value, expr.Col)
        else ColumnStats(name=n.name)
        for n in ir.exprs
    }


@add_source_stats.register(HConcat)
def _(ir: HConcat, stats: StatsCollector) -> None:
    stats.column_stats[ir] = dict(
        itertools.chain.from_iterable(
            stats.column_stats.get(c, {}).items() for c in ir.children
        )
    )


@add_source_stats.register(Union)
def _(ir: Union, stats: StatsCollector) -> None:
    # TODO: Might be able to preserve source statistics
    stats.column_stats[ir] = {name: ColumnStats(name=name) for name in ir.schema}
