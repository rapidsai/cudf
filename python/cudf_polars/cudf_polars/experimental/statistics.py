# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column statistics."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    IR,
    Cache,
    DataFrameScan,
    GroupBy,
    HConcat,
    HStack,
    Join,
    Projection,
    Scan,
    Select,
    Sort,
    Union,
)
from cudf_polars.dsl.traversal import post_traversal
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
    stats: StatsCollector = StatsCollector()
    for node in post_traversal([root]):
        add_source_stats(node, stats, config_options)
    return stats


@add_source_stats.register(IR)
def _(ir: IR, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # Default `add_source_stats` implementation.
    if len(ir.children) == 1:
        (child,) = ir.children
        child_column_stats = stats.column_stats.get(child, {})
        stats.column_stats[ir] = {
            name: child_column_stats.get(name, ColumnStats(name=name))
            for name in ir.schema
        }
    else:
        # Multi-child nodes loose all information by default.
        stats.column_stats[ir] = {name: ColumnStats(name=name) for name in ir.schema}


@add_source_stats.register(Scan)
def _(ir: Scan, stats: StatsCollector, config_options: ConfigOptions) -> None:
    if ir.typ == "parquet":
        stats.column_stats[ir] = {
            name: ColumnStats(
                name=name,
                source_stats=css,
            )
            for name, css in _sample_pq_stats(ir, config_options).items()
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
def _(ir: DataFrameScan, stats: StatsCollector, config_options: ConfigOptions) -> None:
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
def _(ir: Join, stats: StatsCollector, config_options: ConfigOptions) -> None:
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


@add_source_stats.register(GroupBy)
def _(ir: GroupBy, stats: StatsCollector, config_options: ConfigOptions) -> None:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    stats.column_stats[ir] = {
        n.name: child_column_stats.get(n.name, ColumnStats(name=n.name))
        for n in ir.keys
    } | {n.name: ColumnStats(name=n.name) for n in ir.agg_requests}


@add_source_stats.register(HStack)
def _(ir: HStack, stats: StatsCollector, config_options: ConfigOptions) -> None:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    new_cols = {
        n.name: child_column_stats.get(n.value.name, ColumnStats(name=n.name))
        if isinstance(n.value, expr.Col)
        else ColumnStats(name=n.name)
        for n in ir.columns
    }
    stats.column_stats[ir] = child_column_stats | new_cols


@add_source_stats.register(Select)
def _(ir: Select, stats: StatsCollector, config_options: ConfigOptions) -> None:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    stats.column_stats[ir] = {
        n.name: child_column_stats[n.value.name]
        if isinstance(n.value, expr.Col)
        else ColumnStats(name=n.name)
        for n in ir.exprs
    }


@add_source_stats.register(HConcat)
def _(ir: HConcat, stats: StatsCollector, config_options: ConfigOptions) -> None:
    stats.column_stats[ir] = dict(
        itertools.chain.from_iterable(
            stats.column_stats.get(c, {}).items() for c in ir.children
        )
    )


@add_source_stats.register(Union)
def _(ir: Union, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # TODO: Might be able to preserve source statistics
    stats.column_stats[ir] = {name: ColumnStats(name=name) for name in ir.schema}


def _add_source_stats_preserve(
    ir: IR, stats: StatsCollector, config_options: ConfigOptions
) -> None:
    (child,) = ir.children
    stats.column_stats[ir] = {
        name: value
        for name, value in stats.column_stats.get(child, {}).items()
        if name in ir.schema
    }


add_source_stats.register(Cache, _add_source_stats_preserve)
add_source_stats.register(Projection, _add_source_stats_preserve)
add_source_stats.register(Sort, _add_source_stats_preserve)
