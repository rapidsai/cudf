# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column statistics."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, TypeVar

from cudf_polars.dsl.expr import Agg, UnaryFunction
from cudf_polars.dsl.ir import (
    IR,
    DataFrameScan,
    Distinct,
    Filter,
    GroupBy,
    HConcat,
    Join,
    Scan,
    Select,
    Sort,
    Union,
)
from cudf_polars.dsl.traversal import post_traversal, traversal
from cudf_polars.experimental.base import (
    ColumnSourceInfo,
    ColumnStat,
    ColumnStats,
    JoinKey,
    StatsCollector,
)
from cudf_polars.experimental.dispatch import (
    initialize_column_stats,
    update_column_stats,
)
from cudf_polars.experimental.expressions import _SUPPORTED_AGGS
from cudf_polars.experimental.utils import _leaf_column_names
from cudf_polars.utils import conversion
from cudf_polars.utils.cuda_stream import get_cuda_stream

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cudf_polars.dsl.expr import Expr
    from cudf_polars.experimental.base import JoinInfo
    from cudf_polars.typing import Slice as Zlice
    from cudf_polars.utils.config import ConfigOptions, StatsPlanningOptions


def collect_statistics(
    root: IR,
    config_options: ConfigOptions,
) -> StatsCollector:
    """
    Collect column statistics for a query.

    Parameters
    ----------
    root
        Root IR node for collecting column statistics.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A StatsCollector object with populated column statistics.
    """
    assert config_options.executor.name == "streaming", (
        "Only streaming executor is supported in collect_statistics"
    )
    stats_planning = config_options.executor.stats_planning
    need_local_statistics = using_local_statistics(stats_planning)
    if need_local_statistics or stats_planning.use_io_partitioning:
        # Start with base statistics.
        # Here we build an outline of the statistics that will be
        # collected before any real data is sampled. We will not
        # read any Parquet metadata or sample any unique-value
        # statistics during this step.
        # (That said, Polars does it's own metadata sampling
        # before we ever get the logical plan in cudf-polars)
        stats = collect_base_stats(root, config_options)

        # Avoid collecting local statistics unless we are using them.
        if need_local_statistics:
            # Apply PK-FK heuristics.
            # Here we use PK-FK heuristics to estimate the unique count
            # for each join key. We will not do any unique-value sampling
            # during this step. However, we will use Parquet metadata to
            # estimate the row-count for each table source. This metadata
            # is cached in the DataSourceInfo object for each table.
            if stats_planning.use_join_heuristics:
                apply_pkfk_heuristics(stats.join_info)

            # Update statistics for each node.
            # Here we set local row-count and unique-value statistics
            # on each node in the IR graph. We DO perform unique-value
            # sampling during this step. However, we only sample columns
            # that have been marked as needing unique-value statistics
            # during the `collect_base_stats` step. We always sample ALL
            # "marked" columns within the same table source at once.
            for node in post_traversal([root]):
                update_column_stats(node, stats, config_options)

        return stats

    return StatsCollector()


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
    assert config_options.executor.name == "streaming", (
        "Only streaming executor is supported in collect_statistics"
    )
    stats_planning = config_options.executor.stats_planning
    need_local_statistics = using_local_statistics(stats_planning)
    need_join_info = need_local_statistics and stats_planning.use_join_heuristics

    stats: StatsCollector = StatsCollector()
    for node in post_traversal([root]):
        # Initialize column statistics from datasource information
        if need_local_statistics or (
            stats_planning.use_io_partitioning
            and isinstance(node, (Scan, DataFrameScan))
        ):
            stats.column_stats[node] = initialize_column_stats(
                node, stats, config_options
            )
        # Initialize join information
        if need_join_info and isinstance(node, Join):
            initialize_join_info(node, stats)
    return stats


def using_local_statistics(stats_planning: StatsPlanningOptions) -> bool:
    """
    Check if we are using local statistics for query planning.

    Notes
    -----
    This function is used to check if we are using local statistics
    for query-planning purposes. For now, this only returns True
    when `use_reduction_planning=True`. We do not consider `use_io_partitioning`
    here because it only depends on datasource statistics.
    """
    return stats_planning.use_reduction_planning


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
) -> None:
    """Update set of unique-stats columns in datasource."""
    for name in key_names:
        if (column_stats := child_column_stats.get(name)) is not None:
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
    _update_unique_stats_columns(child_column_stats, list(key_names))
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
        # Add key columns to set of unique-stats columns.
        primary_child_stats[p_key.name].source_info.add_unique_stats_column()
        other_child_stats[o_key.name].source_info.add_unique_stats_column()

    return column_stats


@initialize_column_stats.register(GroupBy)
def _(
    ir: GroupBy, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})

    # Update set of source columns we may lazily sample
    _update_unique_stats_columns(child_column_stats, [n.name for n in ir.keys])
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
    return {
        name: ColumnStats(
            name=name,
            children=tuple(stats.column_stats[child][name] for child in ir.children),
            source_info=ColumnSourceInfo(
                *itertools.chain.from_iterable(
                    stats.column_stats[child][name].source_info.table_source_pairs
                    for child in ir.children
                )
            ),
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

    return _extract_dataframescan_stats(ir, config_options)


@initialize_column_stats.register(Select)
def _(
    ir: Select, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    (child,) = ir.children
    column_stats: dict[str, ColumnStats] = {}
    unique_stats_columns: list[str] = []
    child_column_stats = stats.column_stats.get(child, {})
    for ne in ir.exprs:
        if leaf_columns := _leaf_column_names(ne.value):
            # New column is based on 1+ child columns.
            # Inherit the source information from the child columns.
            children = tuple(
                child_column_stats.get(col, ColumnStats(name=col))
                for col in leaf_columns
            )
            column_stats[ne.name] = ColumnStats(
                name=ne.name,
                children=children,
                source_info=ColumnSourceInfo(
                    *itertools.chain.from_iterable(
                        cs.source_info.table_source_pairs for cs in children
                    )
                ),
            )
        else:  # pragma: no cover
            # New column is based on 0 child columns.
            # We don't have any source information to inherit.
            # TODO: Do something smart for a Literal source?
            column_stats[ne.name] = ColumnStats(name=ne.name)

        if any(
            isinstance(expr, UnaryFunction) and expr.name == "unique"
            for expr in traversal([ne.value])
        ):
            # Make sure the leaf column is marked as a unique-stats column.
            unique_stats_columns.extend(list(leaf_columns))

    if unique_stats_columns:
        _update_unique_stats_columns(stats.column_stats[child], unique_stats_columns)

    return column_stats


def known_child_row_counts(ir: IR, stats: StatsCollector) -> list[int]:
    """
    Get all non-null row-count estimates for the children of and IR node.

    Parameters
    ----------
    ir
        IR node to get non-null row-count estimates for.
    stats
        StatsCollector object to get row-count estimates from.

    Returns
    -------
    List of non-null row-count estimates for all children.
    """
    return [
        value
        for child in ir.children
        if (value := stats.row_count[child].value) is not None
    ]


def apply_slice(num_rows: int, zlice: Zlice | None) -> int:
    """Apply a slice to a row-count estimate."""
    if zlice is None:
        return num_rows
    s, e = conversion.from_polars_slice(zlice, num_rows=num_rows)
    return e - s


def apply_predicate_selectivity(
    ir: IR,
    stats: StatsCollector,
    predicate: Expr,
    config_options: ConfigOptions,
) -> None:
    """
    Apply selectivity to a column statistics.

    Parameters
    ----------
    ir
        IR node containing a predicate.
    stats
        The StatsCollector object to update.
    predicate
        The predicate expression.
    config_options
        GPUEngine configuration options.
    """
    assert config_options.executor.name == "streaming", (
        "Only streaming executor is supported in update_column_stats"
    )
    # TODO: Use predicate to generate a better selectivity estimate. Default is 0.8
    selectivity = config_options.executor.stats_planning.default_selectivity
    if selectivity < 1.0 and (row_count := stats.row_count[ir].value) is not None:
        row_count = max(1, int(row_count * selectivity))
        stats.row_count[ir] = ColumnStat[int](row_count)
        for column_stats in stats.column_stats[ir].values():
            if (unique_count := column_stats.unique_count.value) is not None:
                column_stats.unique_count = ColumnStat[int](
                    min(max(1, int(unique_count * selectivity)), row_count)
                )


def copy_child_unique_counts(column_stats_mapping: dict[str, ColumnStats]) -> None:
    """
    Copy unique-count estimates from children to parent.

    Parameters
    ----------
    column_stats_mapping
        Mapping of column names to ColumnStats objects.
    """
    for column_stats in column_stats_mapping.values():
        column_stats.unique_count = ColumnStat[int](
            # Assume we get the maximum child unique-count estimate
            max(
                (
                    cs.unique_count.value
                    for cs in column_stats.children
                    if cs.unique_count.value is not None
                ),
                default=None,
            )
        )


@update_column_stats.register(IR)
def _(ir: IR, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # Default `update_column_stats` implementation.
    # Propagate largest child row-count estimate.
    stats.row_count[ir] = ColumnStat[int](
        max(known_child_row_counts(ir, stats), default=None)
    )

    # Apply slice if relevant.
    # We can also limit the unique-count estimate to the row-count estimate.
    max_unique_count: int | None = None
    if (value := stats.row_count[ir].value) is not None and isinstance(ir, Sort):
        # Apply slice for IR nodes supporting slice pushdown.
        # TODO: Include types other than Sort.
        max_unique_count = apply_slice(value, ir.zlice)
        stats.row_count[ir] = ColumnStat[int](max_unique_count)

    for column_stats in stats.column_stats[ir].values():
        column_stats.unique_count = ColumnStat[int](
            max(
                (
                    min(value, max_unique_count or value)
                    for cs in column_stats.children
                    if (value := cs.unique_count.value) is not None
                ),
                default=None,
            )
        )

    if isinstance(ir, Filter):
        apply_predicate_selectivity(ir, stats, ir.mask.value, config_options)


@update_column_stats.register(DataFrameScan)
def _(
    ir: DataFrameScan,
    stats: StatsCollector,
    config_options: ConfigOptions,
) -> None:
    stream = get_cuda_stream()
    # Use datasource row-count estimate.
    if stats.column_stats[ir]:
        stats.row_count[ir] = next(
            iter(stats.column_stats[ir].values())
        ).source_info.row_count
    else:  # pragma: no cover; We always have stats.column_stats[ir]
        stats.row_count[ir] = ColumnStat[int](None)

    # Update unique-count estimates with sampled statistics
    for column_stats in stats.column_stats[ir].values():
        if column_stats.source_info.implied_unique_count.value is None:
            # We don't have a unique-count estimate, so we need to sample the data.
            source_unique_stats = column_stats.source_info.unique_stats(
                force=False,
            )
            if source_unique_stats.count.value is not None:
                column_stats.unique_count = source_unique_stats.count
        else:
            column_stats.unique_count = column_stats.source_info.implied_unique_count

    stream.synchronize()


@update_column_stats.register(Scan)
def _(
    ir: Scan,
    stats: StatsCollector,
    config_options: ConfigOptions,
) -> None:
    # Use datasource row-count estimate.
    if stats.column_stats[ir]:
        stats.row_count[ir] = next(
            iter(stats.column_stats[ir].values())
        ).source_info.row_count
    else:  # pragma: no cover; We always have stats.column_stats[ir]
        # No column stats available.
        stats.row_count[ir] = ColumnStat[int](None)

    # Account for the n_rows argument.
    if ir.n_rows != -1:
        if (metadata_value := stats.row_count[ir].value) is not None:
            stats.row_count[ir] = ColumnStat[int](min(metadata_value, ir.n_rows))
        else:
            stats.row_count[ir] = ColumnStat[int](ir.n_rows)

    # Update unique-count estimates with estimated and/or sampled statistics
    for column_stats in stats.column_stats[ir].values():
        if column_stats.source_info.implied_unique_count.value is None:
            # We don't have a unique-count estimate, so we need to sample the data.
            source_unique_stats = column_stats.source_info.unique_stats(
                force=False,
            )
            if source_unique_stats.count.value is not None:
                column_stats.unique_count = source_unique_stats.count
            elif (
                unique_fraction := source_unique_stats.fraction.value
            ) is not None and (row_count := stats.row_count[ir].value) is not None:
                column_stats.unique_count = ColumnStat[int](
                    max(1, int(unique_fraction * row_count))
                )
        else:
            column_stats.unique_count = column_stats.source_info.implied_unique_count

    if ir.predicate is not None and ir.n_rows == -1:
        apply_predicate_selectivity(ir, stats, ir.predicate.value, config_options)


@update_column_stats.register(Select)
def _(ir: Select, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # Update statistics for a Select node.

    # Start by copying the child unique-count estimates.
    copy_child_unique_counts(stats.column_stats[ir])

    # Now update the row-count estimate.
    (child,) = ir.children
    child_row_count = stats.row_count.get(child, ColumnStat[int](None)).value
    row_count_estimates: list[int | None] = []
    for ne in ir.exprs:
        child_column_stats = stats.column_stats[ir][ne.name].children
        if isinstance(ne.value, Agg) and ne.value.name in _SUPPORTED_AGGS:
            # This aggregation outputs a single row.
            row_count_estimates.append(1)
            stats.column_stats[ir][ne.name].unique_count = ColumnStat[int](
                value=1, exact=True
            )
        elif (
            len(child_column_stats) == 1
            and any(
                isinstance(expr, UnaryFunction) and expr.name == "unique"
                for expr in traversal([ne.value])
            )
            and (value := child_column_stats[0].unique_count.value) is not None
        ):
            # We are doing a Select(unique) operation.
            row_count_estimates.append(value)
        else:
            # Fallback case - use the child row-count estimate.
            row_count_estimates.append(child_row_count)

    stats.row_count[ir] = ColumnStat[int](
        max((rc for rc in row_count_estimates if rc is not None), default=None),
    )


@update_column_stats.register(Distinct)
@update_column_stats.register(GroupBy)
def _(
    ir: Distinct | GroupBy, stats: StatsCollector, config_options: ConfigOptions
) -> None:
    # Update statistics for a Distinct or GroupBy node.
    (child,) = ir.children
    child_column_stats = stats.column_stats[child]
    child_row_count = stats.row_count[child].value
    key_names = (
        list(ir.subset or ir.schema)
        if isinstance(ir, Distinct)
        else [n.name for n in ir.keys]
    )
    unique_counts = [
        # k will be missing from child_column_stats if it's a literal
        child_column_stats.get(k, ColumnStats(name=k)).unique_count.value
        for k in key_names
    ]
    known_unique_count = sum(c for c in unique_counts if c is not None)
    unknown_unique_count = sum(c is None for c in unique_counts)
    if unknown_unique_count > 0:
        # Use the child row-count to be conservative.
        # TODO: Should we use a different heuristic here? For example,
        # we could assume each unknown key introduces a factor of 3.
        stats.row_count[ir] = ColumnStat[int](child_row_count)
    else:
        unique_count = known_unique_count
        if child_row_count is not None:
            # Don't allow the unique-count to exceed the child row-count.
            unique_count = min(child_row_count, unique_count)
        stats.row_count[ir] = ColumnStat[int](unique_count)

    copy_child_unique_counts(stats.column_stats[ir])


@update_column_stats.register(Join)
def _(ir: Join, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # Apply basic join-cardinality estimation.
    child_row_counts = known_child_row_counts(ir, stats)
    if len(child_row_counts) == 2:
        # Both children have row-count estimates.

        # Use the PK-FK unique-count estimate for the join key.
        # Otherwise, use the maximum unique-count estimate from the children.
        unique_count_estimate = max(
            # Join-based estimate (higher priority).
            [
                u.implied_unique_count
                for u in stats.join_info.join_map.get(ir, [])
                if u.implied_unique_count is not None
            ],
            default=None,
        )
        # TODO: Use local unique-count statistics if the implied unique-count
        # estimates are missing. This never happens for now, but it will happen
        # if/when we add a config option to disable PK-FK heuristics.

        # Calculate the output row-count estimate.
        left_rows, right_rows = child_row_counts
        if unique_count_estimate is not None:
            stats.row_count[ir] = ColumnStat[int](
                max(1, (left_rows * right_rows) // unique_count_estimate)
            )
        else:  # pragma: no cover; We always have a unique-count estimate (for now).
            stats.row_count[ir] = ColumnStat[int](max((1, left_rows, right_rows)))
    else:
        # One or more children have an unknown row-count estimate.
        stats.row_count[ir] = ColumnStat[int](None)

    copy_child_unique_counts(stats.column_stats[ir])


@update_column_stats.register(Union)
def _(ir: Union, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # Add up child row-count estimates.
    row_counts = known_child_row_counts(ir, stats)
    stats.row_count[ir] = ColumnStat[int](sum(row_counts) or None)
    # Add up unique counts (NOTE: This is probably very conservative).
    for column_stats in stats.column_stats.get(ir, {}).values():
        column_stats.unique_count = ColumnStat[int](
            sum(
                (
                    cs.unique_count.value
                    for cs in column_stats.children
                    if cs.unique_count.value is not None
                ),
            )
            or None
        )
