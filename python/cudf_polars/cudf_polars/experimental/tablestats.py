# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for cardinality estimation."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

import polars as pl

import pylibcudf as plc

from cudf_polars.dsl import expr
from cudf_polars.dsl.ir import (
    Cache,
    DataFrameScan,
    Distinct,
    Filter,
    GroupBy,
    HConcat,
    HStack,
    Join,
    Projection,
    Scan,
    Select,
    Slice,
    Sort,
    Union,
)
from cudf_polars.dsl.traversal import post_traversal
from cudf_polars.utils import conversion

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.typing import Slice as Zlice


class ColumnStats:  # noqa: D101
    def __init__(
        self,
        unique_count: int | None,
        source_rows: int | None = None,
        name: str | None = None,
    ) -> None:
        self.unique_count = unique_count
        self.source_rows = source_rows
        self.name = name

    def update(self, unique_count: int | None) -> None:  # noqa: D102
        if self.unique_count is None or (
            unique_count is not None and unique_count < self.unique_count
        ):
            self.unique_count = unique_count

    def __repr__(self) -> str:  # noqa: D105
        if self.name is not None:
            return f"ColumnStats({self.name}, {self.unique_count})"
        else:
            return super().__repr__()


class JoinKey:  # noqa: D101
    def __init__(self, *columns: ColumnStats) -> None:
        self.source_rows = max(
            (c.source_rows for c in columns if c.source_rows is not None), default=None
        )
        self.unique_count: int | None = None
        self.name = "-".join(str(c.name) for c in columns)
        self.columns = columns

    def __repr__(self) -> str:  # noqa: D105
        return f"JoinKey({self.name}, {self.unique_count})"


class StatsCollector:  # noqa: D101
    def __init__(self) -> None:
        self.estimated_row_counts: dict[IR, int] = {}
        self.column_statistics: dict[IR, dict[str, ColumnStats]] = {}
        self.key_joins: MutableMapping[JoinKey, set[JoinKey]] = defaultdict(
            set[JoinKey]
        )
        self.col_joins: MutableMapping[ColumnStats, set[ColumnStats]] = defaultdict(
            set[ColumnStats]
        )
        self.joins: dict[IR, list[JoinKey]] = {}


def collect_basic_statistics(root: IR) -> StatsCollector:  # noqa: D103
    stats = StatsCollector()
    for node in post_traversal([root]):
        if isinstance(node, Scan):
            if node.typ == "parquet":
                # TODO: use pylibcudf, get remaining metadata statistics
                nrows = (
                    pl.scan_parquet(node.paths).select(pl.len()).collect().item(0, 0)
                )
                stats.column_statistics[node] = {
                    name: ColumnStats(None, nrows, name=name) for name in node.schema
                }
                stats.estimated_row_counts[node] = nrows
            else:
                raise NotImplementedError("No table stats for csv/json")
        elif isinstance(node, DataFrameScan):
            nrows = node.df.height()
            stats.column_statistics[node] = {
                name: ColumnStats(None, nrows, name=name) for name in node.schema
            }
            stats.estimated_row_counts[node] = node.df.height()
        elif isinstance(node, Join):
            assert node.options[0] != "Cross"  # no support yet
            left, right = node.children
            left_keys = [stats.column_statistics[left][n.name] for n in node.left_on]
            right_keys = [stats.column_statistics[right][n.name] for n in node.right_on]
            lkey = JoinKey(*left_keys)
            rkey = JoinKey(*right_keys)
            stats.key_joins[lkey].add(rkey)
            stats.key_joins[rkey].add(lkey)
            stats.joins[node] = [lkey, rkey]
            for u, v in zip(left_keys, right_keys, strict=True):
                stats.col_joins[u].add(v)
                stats.col_joins[v].add(u)
            # More complicated than this if coalescing is off
            kstats = {
                n.name: stats.column_statistics[left][n.name] for n in node.left_on
            }
            jstats = {
                name: stats.column_statistics[left][name]
                for name in left.schema
                if name not in kstats
            }
            suffix = node.options[3]
            jstats |= {
                name
                if name not in jstats
                else f"{name}{suffix}": stats.column_statistics[right][name]
                for name in right.schema
                if name not in kstats
            }
            stats.column_statistics[node] = kstats | jstats
        elif isinstance(node, GroupBy):
            (child,) = node.children
            stats.column_statistics[node] = {
                n.name: stats.column_statistics[child][n.name] for n in node.keys
            } | {n.name: ColumnStats(None, name=n.name) for n in node.agg_requests}
        elif isinstance(node, Projection):
            stats.column_statistics[node] = stats.column_statistics[node.children[0]]
        elif isinstance(node, HStack):
            (child,) = node.children
            new_cols = {
                n.name: stats.column_statistics[child][n.value.name]
                if isinstance(n.value, expr.Col)
                else ColumnStats(None, name=n.name)
                for n in node.columns
            }
            stats.column_statistics[node] = stats.column_statistics[child] | new_cols
        elif isinstance(node, Select):
            (child,) = node.children
            stats.column_statistics[node] = {
                n.name: stats.column_statistics[child][n.value.name]
                if isinstance(n.value, expr.Col)
                else ColumnStats(None, name=n.name)
                for n in node.exprs
            }
        elif isinstance(node, HConcat):
            stats.column_statistics[node] = dict(
                itertools.chain.from_iterable(
                    stats.column_statistics[c].items() for c in node.children
                )
            )
        elif isinstance(node, Union):
            stats.column_statistics[node] = {
                name: ColumnStats(None, name=name) for name in node.schema
            }
        elif isinstance(node, (Sort, Distinct, Filter, Cache)):
            stats.column_statistics[node] = stats.column_statistics[node.children[0]]
        else:
            raise NotImplementedError(f"Unhandled node type {type(node).__name__}")
    return stats


T = TypeVar("T")


def find_equivalence_sets(  # noqa: D103
    joins: Mapping[T, set[T]],
) -> list[set[T]]:
    seen = set()
    components = []
    for v in joins:
        if v not in seen:
            cluster = {v}
            stack = [v]
            while stack:
                node = stack.pop()
                for n in joins[node]:
                    if n not in cluster:
                        cluster.add(n)
                        stack.append(n)
            components.append(cluster)
            seen.update(cluster)
    return components


def estimate_unique_counts(ir: IR) -> StatsCollector:  # noqa: D103
    # This applies the foreign-key -- primary-key matching scheme of
    # https://blobs.duckdb.org/papers/tom-ebergen-msc-thesis-join-order-optimization-with-almost-no-statistics.pdf
    # See section 3.2
    stats = collect_basic_statistics(ir)
    # We separately track equivalence sets of "joins" (which might use multiple keys)
    # and columns. This way we can deduce cardinality estimates for
    # joins separately from the underlying columns we join on.
    key_clusters = find_equivalence_sets(stats.key_joins)
    for keys in key_clusters:
        unique_count = max(
            (c.unique_count for c in keys if c.unique_count is not None),
            default=min(
                (c.source_rows for c in keys if c.source_rows is not None),
                default=None,
            ),
        )
        for key in keys:
            key.unique_count = unique_count
    col_clusters = find_equivalence_sets(stats.col_joins)
    for cols in col_clusters:
        unique_count = max(
            (c.unique_count for c in cols if c.unique_count is not None),
            default=min(
                (c.source_rows for c in cols if c.source_rows is not None),
                default=None,
            ),
        )
        for col in cols:
            col.update(unique_count)
    return stats


def apply_slice(rows: int, zlice: Zlice | None) -> int:  # noqa: D103
    if zlice is None:
        return rows
    s, e = conversion.from_polars_slice(zlice, num_rows=rows)
    return e - s


def collect_statistics(ir: IR) -> StatsCollector:  # noqa: D103
    stats = estimate_unique_counts(ir)
    rows = stats.estimated_row_counts
    for node in post_traversal([ir]):
        colstats = stats.column_statistics[node]
        if node in rows:
            if isinstance(node, Scan):
                nrows = rows[node] if node.n_rows == -1 else node.n_rows
                if node.predicate is not None:
                    # TODO: multiple filter expressions?
                    selectivity = 0.8
                    if (
                        isinstance((pred := node.predicate.value), expr.BinOp)
                        and pred.op is plc.binaryop.BinaryOperator.EQUAL
                    ):
                        try:
                            (col,) = (
                                c for c in pred.children if isinstance(c, expr.Col)
                            )
                            # Equality between column and something, assume uniformly distributed
                            if (
                                col.name in stats.column_statistics[node]
                                and (
                                    unique_count := stats.column_statistics[node][
                                        col.name
                                    ].unique_count
                                )
                                is not None
                            ):
                                selectivity = 1 / unique_count
                        except ValueError:
                            pass
                    rows[node] = max(1, int(nrows * selectivity))
                else:
                    rows[node] = nrows
        elif isinstance(node, GroupBy):
            keycols = [k.name for k in node.keys]
            counts = [colstats[k].unique_count for k in keycols]
            # cartesian product is probably a bad estimate, use sum
            # This is a total guess though
            # Maybe use cartesian product, correct for correlation and
            # occupancy problem estimates?
            # https://probabilityandstats.wordpress.com/2010/03/27/the-occupancy-problem/
            known_count = sum(c for c in counts if c is not None)
            unknown = sum(c is None for c in counts)
            if unknown == len(counts):
                # Total guess
                rows[node] = apply_slice(
                    max(1, rows[node.children[0]] // 100), node.zlice
                )
            else:
                # Guess each additional key introduces a factor of 3
                rows[node] = apply_slice(known_count * 3**unknown, node.zlice)
        elif isinstance(node, Join):
            left, right = node.children
            left_rows = rows[left]
            right_rows = rows[right]
            unique = max(
                (
                    u.unique_count
                    for u in stats.joins[node]
                    if u.unique_count is not None
                ),
                default=None,
            )
            if unique is not None:
                rows[node] = apply_slice(
                    max(1, (left_rows * right_rows) // unique), node.options[2]
                )
            else:
                rows[node] = apply_slice(
                    max((1, left_rows, right_rows)), node.options[2]
                )
        elif isinstance(node, Filter):
            # TODO: multiple filter expressions?
            rows[node] = max(1, int(rows[node.children[0]] * 0.8))
        elif isinstance(node, (Select, Projection, HStack, Cache)):
            # TODO: Select with all aggregates should produce a single
            # row as output.
            rows[node] = rows[node.children[0]]
        elif isinstance(node, Sort):
            rows[node] = apply_slice(rows[node.children[0]], node.zlice)
        elif isinstance(node, (Scan, DataFrameScan)):
            raise AssertionError("Should have already handled this")
        elif isinstance(node, Union):
            rows[node] = apply_slice(sum(rows[c] for c in node.children), node.zlice)
        elif isinstance(node, HConcat):
            rows[node] = max(rows[c] for c in node.children)
        elif isinstance(node, Slice):
            rows[node] = apply_slice(rows[node.children[0]], (node.offset, node.length))
    return stats
