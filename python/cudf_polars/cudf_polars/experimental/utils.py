# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition utilities."""

from __future__ import annotations

import operator
import warnings
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING

from cudf_polars.dsl.expr import Col, Expr, GroupedRollingWindow, UnaryFunction
from cudf_polars.dsl.ir import Union
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import ColumnStat, PartitionInfo

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.base import ColumnStats
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions


def _concat(*dfs: DataFrame, context: IRExecutionContext) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return dfs[0] if len(dfs) == 1 else Union.do_evaluate(None, *dfs, context=context)


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


def _dynamic_planning_on(config_options: ConfigOptions) -> bool:
    """Check if dynamic planning is enabled for rapidsmpf runtime."""
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node'"
    )

    return (
        config_options.executor.runtime == "rapidsmpf"
        and config_options.executor.dynamic_planning is not None
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

    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_sub_network'"
    )
    rapidsmpf_engine = config_options.executor.runtime == "rapidsmpf"

    # Lower children
    lowered_children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Ensure all children are single-partitioned
    children = []
    inform = False
    for c in lowered_children:
        child = c
        if multi_partitioned := partition_info[c].count > 1:
            inform = True
        if multi_partitioned or rapidsmpf_engine:
            # Fall-back logic
            child = Repartition(child.schema, child)
            partition_info[child] = PartitionInfo(count=1)
        children.append(child)

    if inform and msg:
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


def _get_unique_fractions(
    column_names: Sequence[str],
    user_unique_fractions: dict[str, float],
    *,
    row_count: ColumnStat[int] | None = None,
    column_stats: dict[str, ColumnStats] | None = None,
) -> dict[str, float]:
    """
    Return unique-fraction statistics subset.

    Parameters
    ----------
    column_names
        The column names to get unique-fractions for.
    user_unique_fractions
        The user-provided unique-fraction dictionary.
    row_count
        Row-count statistics. This will be None if
        statistics planning is not enabled.
    column_stats
        The column statistics. This will be None if
        statistics planning is not enabled.

    Returns
    -------
    unique_fractions
        The final unique-fraction dictionary.
    """
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
