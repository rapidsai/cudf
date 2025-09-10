# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition utilities."""

from __future__ import annotations

import math
import operator
import warnings
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl.expr import Col, Expr, GroupedRollingWindow, UnaryFunction
from cudf_polars.dsl.ir import Union
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import ColumnStat, PartitionInfo

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import ColumnStats, StatsCollector
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions


def _concat(*dfs: DataFrame) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return Union.do_evaluate(None, *dfs)


def _fallback_inform(msg: str, config_options: ConfigOptions) -> None:
    """Inform the user of single-partition fallback."""
    # Satisfy type checking
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in '_fallback_inform'"
    )

    match fallback_mode := config_options.executor.fallback_mode:
        case "warn":
            warnings.warn(msg, stacklevel=2)
        case "raise":
            raise NotImplementedError(msg)
        case "silent":
            pass
        case _:  # pragma: no cover; Should never get here.
            raise ValueError(
                f"{fallback_mode} is not a supported 'fallback_mode' "
                "option. Please use 'warn', 'raise', or 'silent'."
            )


def _lower_ir_fallback(
    ir: IR,
    rec: LowerIRTransformer,
    *,
    msg: str | None = None,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Catch-all single-partition lowering logic.
    # If any children contain multiple partitions,
    # those children will be collapsed with `Repartition`.
    from cudf_polars.experimental.repartition import Repartition

    # Lower children
    lowered_children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Ensure all children are single-partitioned
    children = []
    fallback = False
    for c in lowered_children:
        child = c
        if partition_info[c].count > 1:
            # Fall-back logic
            fallback = True
            child = Repartition(child.schema, child)
            partition_info[child] = PartitionInfo(count=1)
        children.append(child)

    if fallback and msg:
        # Warn/raise the user if any children were collapsed
        # and the "fallback_mode" configuration is not "silent"
        _fallback_inform(msg, rec.state["config_options"])

    # Reconstruct and return
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=1)
    return new_node, partition_info


def _leaf_column_names(expr: Expr) -> tuple[str, ...]:
    """Find the leaf column names of an expression."""
    if expr.children:
        return tuple(
            chain.from_iterable(_leaf_column_names(child) for child in expr.children)
        )
    elif isinstance(expr, Col):
        return (expr.name,)
    else:
        return ()


def _estimate_ideal_partition_count(
    ir: IR,
    stats: StatsCollector,
    config_options: ConfigOptions,
) -> int | None:
    """Estimate the ideal number of partitions for a query node."""
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in '_estimate_ideal_partition_count'"
    )
    row_count = stats.row_count.get(ir, ColumnStat[int](None))
    if (
        row_count.value is None
        or config_options.executor.statistics_planning_options.enable
    ):
        return None

    size = 0
    column_stats = stats.column_stats.get(ir, {})
    for col, dtype in ir.schema.items():
        if col in column_stats:
            # if (itemsize := column_stats[col].source_info.storage_size.value) is None:
            try:
                itemsize = plc.types.size_of(dtype.plc)
            except RuntimeError:
                # Pylibcudf will raise a RuntimeError for non fixed-width types.
                # Default to 32 bytes for these cases. This is basically a
                # complete guess, but may be better than nothing.
                itemsize = 32
            size += row_count.value * itemsize
        else:
            return None

    target_partition_size = config_options.executor.target_partition_size
    return max(1, math.ceil(size / target_partition_size))


def _get_unique_fractions(
    column_names: Sequence[str],
    user_unique_fractions: dict[str, float],
    *,
    row_count: ColumnStat[int] | None = None,
    column_stats: dict[str, ColumnStats] | None = None,
) -> dict[str, float]:
    """Return unique-fraction statistics subset."""
    unique_fractions: dict[str, float] = {}
    column_stats = column_stats or {}
    row_count = row_count or ColumnStat[int](None)
    if isinstance(row_count.value, int) and row_count.value > 0:
        for c in set(column_names).intersection(column_stats):
            if (unique_count := column_stats[c].unique_count.value) is not None:
                # Use unique_count_estimate (if available)
                unique_fractions[c] = max(
                    min(1.0, unique_count / row_count.value),
                    0.00001,
                )

    # # Debug write to file
    # with open("unique_fractions.txt", "a") as f:
    #     f.write(f"{unique_fractions}\n")

    # Update with user-provided unique-fractions
    unique_fractions.update(
        {
            c: max(min(f, 1.0), 0.00001)
            for c, f in user_unique_fractions.items()
            if c in column_names
        }
    )
    return unique_fractions


def _contains_over(exprs: Sequence[Expr]) -> bool:
    """Return True if any expression in 'exprs' contains an over(...) (ie. GroupedRollingWindow)."""
    return any(isinstance(e, GroupedRollingWindow) for e in traversal(exprs))


def _contains_unsupported_fill_strategy(exprs: Sequence[Expr]) -> bool:
    for e in traversal(exprs):
        if (
            isinstance(e, UnaryFunction)
            and e.name == "fill_null_with_strategy"
            and e.options[0] not in ("zero", "one")
        ):
            return True
    return False
