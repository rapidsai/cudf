# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column statistics."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    IR,
    DataFrameScan,
    Distinct,
    GroupBy,
    HConcat,
    HStack,
    Join,
    Scan,
    Select,
    Union,
)
from cudf_polars.dsl.traversal import post_traversal
from cudf_polars.experimental.base import (
    ColumnStats,
    StatsCollector,
)
from cudf_polars.experimental.dispatch import extract_base_stats

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_polars.utils.config import ConfigOptions


def collect_base_stats(root: IR, config_options: ConfigOptions) -> StatsCollector:
    """
    Collect base datasource statistics.

    Parameters
    ----------
    root
        Root IR node for collecting base datasource statistics.
    config_options
        GPUEngine configuration options.
    """
    stats: StatsCollector = StatsCollector()
    for node in post_traversal([root]):
        stats.column_stats[node] = extract_base_stats(node, stats, config_options)
    return stats


def _update_unique_stats_columns(
    child_column_stats: dict[str, ColumnStats],
    key_names: Sequence[str],
    config_options: ConfigOptions,
) -> None:
    """Update set of unique-stats columns in datasource."""
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'add_source_stats'"
    )
    unique_fraction = config_options.executor.unique_fraction
    for name in key_names:
        if (
            name not in unique_fraction
            and (column_stats := child_column_stats.get(name)) is not None
            and (source_stats := column_stats.source_info) is not None
        ):
            source_stats.add_unique_stats_column(column_stats.source_name or name)


@extract_base_stats.register(IR)
def _default_extract_base_stats(
    ir: IR, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # Default `extract_base_stats` implementation.
    if len(ir.children) == 1:
        (child,) = ir.children
        child_column_stats = stats.column_stats.get(child, {})
        return {
            name: child_column_stats.get(name, ColumnStats(name=name))
            for name in ir.schema
        }
    else:  # pragma: no cover
        # Multi-child nodes loose all information by default.
        return {name: ColumnStats(name=name) for name in ir.schema}


@extract_base_stats.register(Distinct)
def _(
    ir: Distinct, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # Use default extract_base_stats after updating
    # the known unique-stats columns.
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    key_names = ir.subset or ir.schema
    _update_unique_stats_columns(child_column_stats, list(key_names), config_options)
    return _default_extract_base_stats(ir, stats, config_options)


@extract_base_stats.register(Join)
def _(
    ir: Join, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    left, right = ir.children
    how = ir.options[0]
    suffix = ir.options[3]
    primary, other = (right, left) if how == "Right" else (left, right)
    primary_child_stats = stats.column_stats.get(primary, {})
    other_child_stats = stats.column_stats.get(other, {})

    # Build output column statistics
    column_stats: dict[str, ColumnStats] = {}
    for name in ir.schema:
        if name in primary.schema:
            # "Primary" child stats take preference.
            column_stats[name] = primary_child_stats.get(name, ColumnStats(name=name))
        elif name in other.schema:
            # "Other" column stats apply to everything else.
            column_stats[name] = other_child_stats.get(name, ColumnStats(name=name))
        else:
            # If the column name was not in either child table,
            # a suffix was probably added to a column in "other".
            _name = name.removesuffix(suffix)
            column_stats[name] = other_child_stats.get(
                _name, ColumnStats(name=name)
            ).rename(name)

    return column_stats


@extract_base_stats.register(GroupBy)
def _(
    ir: GroupBy, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})

    # Update set of source columns we may lazily sample
    _update_unique_stats_columns(
        child_column_stats, [n.name for n in ir.keys], config_options
    )

    return {
        n.name: child_column_stats.get(n.name, ColumnStats(name=n.name))
        for n in ir.keys
    } | {n.name: ColumnStats(name=n.name) for n in ir.agg_requests}


@extract_base_stats.register(HStack)
def _(
    ir: HStack, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    new_cols = {
        n.name: child_column_stats.get(n.value.name, ColumnStats(name=n.name))
        if isinstance(n.value, expr.Col)
        else ColumnStats(name=n.name)
        for n in ir.columns
    }
    return child_column_stats | new_cols


@extract_base_stats.register(Select)
def _(
    ir: Select, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    return {
        n.name: child_column_stats.get(n.value.name, ColumnStats(name=n.name))
        if isinstance(n.value, expr.Col)
        else ColumnStats(name=n.name)
        for n in ir.exprs
    }


@extract_base_stats.register(HConcat)
def _(
    ir: HConcat, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    return dict(
        itertools.chain.from_iterable(
            stats.column_stats.get(c, {}).items() for c in ir.children
        )
    )


@extract_base_stats.register(Union)
def _(
    ir: Union, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # TODO: We can preserve matching source statistics
    return {name: ColumnStats(name=name) for name in ir.schema}


@extract_base_stats.register(Scan)
def _(
    ir: Scan, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    from cudf_polars.experimental.io import _extract_scan_stats

    return _extract_scan_stats(ir, config_options)


@extract_base_stats.register(DataFrameScan)
def _(
    ir: DataFrameScan, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    from cudf_polars.experimental.io import _extract_dataframescan_stats

    return _extract_dataframescan_stats(ir)
