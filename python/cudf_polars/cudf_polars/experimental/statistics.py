# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column statistics."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import (
    IR,
    DataFrameScan,
    Distinct,
    GroupBy,
    HConcat,
    Join,
    Scan,
    Union,
)
from cudf_polars.dsl.traversal import post_traversal
from cudf_polars.experimental.base import (
    ColumnStats,
    StatsCollector,
)
from cudf_polars.experimental.dispatch import initialize_column_stats

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

    Returns
    -------
    A new StatsCollector object with populated datasource statistics.
    """
    stats: StatsCollector = StatsCollector()
    for node in post_traversal([root]):
        stats.column_stats[node] = initialize_column_stats(node, stats, config_options)
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
        ):
            column_stats.source_info.add_unique_stats_column()


@initialize_column_stats.register(IR)
def _default_initialize_column_stats(
    ir: IR, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # Default `initialize_column_stats` implementation.
    if len(ir.children) == 1:
        (child,) = ir.children
        child_column_stats = stats.column_stats.get(child, {})
        return {
            name: child_column_stats.get(name, ColumnStats(name=name)).new_parent()
            for name in ir.schema
        }
    else:  # pragma: no cover
        # Multi-child nodes loose all information by default.
        return {name: ColumnStats(name=name) for name in ir.schema}


@initialize_column_stats.register(Distinct)
def _(
    ir: Distinct, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # Use default initialize_column_stats after updating
    # the known unique-stats columns.
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    key_names = ir.subset or ir.schema
    _update_unique_stats_columns(child_column_stats, list(key_names), config_options)
    return _default_initialize_column_stats(ir, stats, config_options)


@initialize_column_stats.register(Join)
def _(
    ir: Join, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # Copy column statistics from both the left and right children.
    # Special cases to consider:
    #   - If a column name appears in both sides of the join,
    #     we take it from the "primary" column (right for "Right"
    #     joins, left for all other joins).
    #   - If a column name doesn't appear in either child, it
    #     corresponds to a non-"primary" column with a suffix.

    children, on = ir.children, (ir.left_on, ir.right_on)
    how = ir.options[0]
    suffix = ir.options[3]
    if how == "Right":
        children, on = children[::-1], on[::-1]
    primary, other = children
    primary_child_stats = stats.column_stats.get(primary, {})
    other_child_stats = stats.column_stats.get(other, {})

    # Build output column statistics
    column_stats: dict[str, ColumnStats] = {}
    for name in ir.schema:
        if name in primary.schema:
            # "Primary" child stats take preference.
            column_stats[name] = primary_child_stats[name].new_parent()
        elif name in other.schema:
            # "Other" column stats apply to everything else.
            column_stats[name] = other_child_stats[name].new_parent()
        else:
            # If the column name was not in either child table,
            # a suffix was added to a column in "other".
            _name = name.removesuffix(suffix)
            column_stats[name] = other_child_stats[_name].new_parent(name=name)

    # Update children
    for p_key, o_key in zip(*on, strict=True):
        column_stats[p_key.name].children = (
            primary_child_stats[p_key.name],
            other_child_stats[o_key.name],
        )

    return column_stats


@initialize_column_stats.register(GroupBy)
def _(
    ir: GroupBy, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})

    # Update set of source columns we may lazily sample
    _update_unique_stats_columns(
        child_column_stats, [n.name for n in ir.keys], config_options
    )
    return _default_initialize_column_stats(ir, stats, config_options)


@initialize_column_stats.register(HConcat)
def _(
    ir: HConcat, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    child_column_stats = dict(
        itertools.chain.from_iterable(
            stats.column_stats.get(c, {}).items() for c in ir.children
        )
    )
    return {
        name: child_column_stats.get(name, ColumnStats(name=name)).new_parent()
        for name in ir.schema
    }


@initialize_column_stats.register(Union)
def _(
    ir: IR, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # Union looses source information for now.
    return {
        name: ColumnStats(
            name=name,
            children=tuple(stats.column_stats[child][name] for child in ir.children),
        )
        for name in ir.schema
    }


@initialize_column_stats.register(Scan)
def _(
    ir: Scan, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    from cudf_polars.experimental.io import _extract_scan_stats

    return _extract_scan_stats(ir, config_options)


@initialize_column_stats.register(DataFrameScan)
def _(
    ir: DataFrameScan, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    from cudf_polars.experimental.io import _extract_dataframescan_stats

    return _extract_dataframescan_stats(ir)
