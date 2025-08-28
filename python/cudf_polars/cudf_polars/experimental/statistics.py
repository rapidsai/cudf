# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column statistics."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, TypeVar

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
    ColumnStat,
    ColumnStats,
    JoinKey,
    StatsCollector,
)
from cudf_polars.experimental.dispatch import initialize_column_stats

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cudf_polars.experimental.base import JoinInfo
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

    Notes
    -----
    This function initializes the ``StatsCollector`` object
    with the base datasource statistics. The goal is to build an
    outline of the statistics that will be collected before any
    real data is sampled.
    """
    stats: StatsCollector = StatsCollector()
    for node in post_traversal([root]):
        # Initialize column statistics from datasource information
        stats.column_stats[node] = initialize_column_stats(node, stats, config_options)
        # Initialize join information
        if isinstance(node, Join):
            initialize_join_info(node, stats)
    return stats


def initialize_join_info(node: Join, stats: StatsCollector) -> None:
    """
    Initialize join information for the given node.

    Parameters
    ----------
    node
        Join node to initialize join-key information for.
    stats
        StatsCollector object to update.

    Notes
    -----
    This function updates ``stats.join_info``.
    """
    left, right = node.children
    join_info = stats.join_info
    right_keys = [stats.column_stats[right][n.name] for n in node.right_on]
    left_keys = [stats.column_stats[left][n.name] for n in node.left_on]
    lkey = JoinKey(*right_keys)
    rkey = JoinKey(*left_keys)
    join_info.key_map[lkey].add(rkey)
    join_info.key_map[rkey].add(lkey)
    join_info.join_map[node] = [lkey, rkey]
    for u, v in zip(left_keys, right_keys, strict=True):
        join_info.column_map[u].add(v)
        join_info.column_map[v].add(u)


T = TypeVar("T")


def find_equivalence_sets(join_map: Mapping[T, set[T]]) -> list[set[T]]:
    """
    Find equivalence sets in a join-key mapping.

    Parameters
    ----------
    join_map
        Joined key or column mapping to find equivalence sets in.

    Returns
    -------
    List of equivalence sets.

    Notes
    -----
    This function is used by ``apply_pkfk_heuristics``.
    """
    seen = set()
    components = []
    for v in join_map:
        if v not in seen:
            cluster = {v}
            stack = [v]
            while stack:
                node = stack.pop()
                for n in join_map[node]:
                    if n not in cluster:
                        cluster.add(n)
                        stack.append(n)
            components.append(cluster)
            seen.update(cluster)
    return components


def apply_pkfk_heuristics(join_info: JoinInfo) -> None:
    """
    Apply PK-FK unique-count heuristics to join keys.

    Parameters
    ----------
    join_info
        Join information to apply PK-FK heuristics to.

    Notes
    -----
    This function modifies the ``JoinKey`` objects being tracked
    in ``StatsCollector.join_info`` using PK-FK heuristics to
    estimate the "implied" unique-value count. This function also
    modifies the inderlying ``ColumnStats`` objects included in
    a join key.
    """
    # This applies the PK-FK matching scheme of
    # https://blobs.duckdb.org/papers/tom-ebergen-msc-thesis-join-order-optimization-with-almost-no-statistics.pdf
    # See section 3.2
    for keys in find_equivalence_sets(join_info.key_map):
        implied_unique_count = max(
            (
                c.implied_unique_count
                for c in keys
                if c.implied_unique_count is not None
            ),
            # Default unique-count estimate is the minimum source row count
            default=min(
                (c.source_row_count for c in keys if c.source_row_count is not None),
                default=None,
            ),
        )
        for key in keys:
            # Update unique-count estimate for each join key
            key.implied_unique_count = implied_unique_count

    # We separately apply PK-FK heuristics to individual columns so
    # that we can update ColumnStats.source_info.implied_unique_count
    # and use the per-column information elsewhere in the query plan.
    for cols in find_equivalence_sets(join_info.column_map):
        unique_count = max(
            (
                cs.source_info.implied_unique_count.value
                for cs in cols
                if cs.source_info.implied_unique_count.value is not None
            ),
            default=min(
                (
                    cs.source_info.row_count.value
                    for cs in cols
                    if cs.source_info.row_count.value is not None
                ),
                default=None,
            ),
        )
        for cs in cols:
            cs.source_info.implied_unique_count = ColumnStat[int](unique_count)


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
