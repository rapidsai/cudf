# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition utilities."""

from __future__ import annotations

import operator
import warnings
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING

from cudf_polars.dsl.expr import Col, Expr, GroupedWindow, UnaryFunction
from cudf_polars.dsl.ir import Union
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import PartitionInfo

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


def _concat(*dfs: DataFrame, context: IRExecutionContext) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return dfs[0] if len(dfs) == 1 else Union.do_evaluate(None, *dfs, context=context)


def _fallback_inform(
    msg: str, config_options: ConfigOptions[StreamingExecutor]
) -> None:
    """Inform the user of single-partition fallback."""
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


def _dynamic_planning_on(config_options: ConfigOptions[StreamingExecutor]) -> bool:
    """Check if dynamic planning is enabled for rapidsmpf runtime."""
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
    from cudf_polars.experimental.select import _inline_hstack_false

    config_options = rec.state["config_options"]
    rapidsmpf_engine = config_options.executor.runtime == "rapidsmpf"

    # Make sure we avoid mixed-length columns in intermediate TableChunks.
    ir = _inline_hstack_false(ir)

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
) -> dict[str, float]:
    """
    Return unique-fraction statistics subset.

    Parameters
    ----------
    column_names
        The column names to get unique-fractions for.
    user_unique_fractions
        The user-provided unique-fraction dictionary.

    Returns
    -------
    unique_fractions
        The final unique-fraction dictionary filtered to column_names.
    """
    return {
        c: max(min(f, 1.0), 0.00001)
        for c, f in user_unique_fractions.items()
        if c in column_names
    }


def _contains_over(exprs: Sequence[Expr]) -> bool:
    """Return True if any expression contains a window expression."""
    return any(isinstance(e, GroupedWindow) for e in traversal(exprs))


def _extract_over_shuffle_indices(
    exprs: Sequence[Expr], child_schema: Schema
) -> tuple[int, ...] | None:
    """
    Return column indices for hash-shuffling a window over() operation.

    Returns
    -------
    tuple[int, ...]
        Empty: no GroupedWindow found; plain chunkwise is correct.
        Non-empty: all GroupedWindow share the same partition-by keys;
        values are indices of those keys in ``child_schema``.
    None
        Multiple distinct partition-by key sets, or any partition-by
        expression is not a plain column reference; not supported for
        multi-partition streaming (caller should fall back to
        single-partition).
    """
    seen_key_sets: set[frozenset[str]] = set()
    for node in traversal(exprs):
        if isinstance(node, GroupedWindow):
            by_children = node.children[: node.by_count]
            # TODO: support non-Col partition-by expressions using
            # ShuffleManager.insert_hash_with_keys (rapidsai/cudf#21692).
            if not all(isinstance(c, Col) for c in by_children):
                return None
            seen_key_sets.add(frozenset(c.name for c in by_children))  # type: ignore[attr-defined]
    if not seen_key_sets:
        return ()
    if len(seen_key_sets) > 1:
        return None
    schema_keys = list(child_schema.keys())
    try:
        return tuple(schema_keys.index(n) for n in next(iter(seen_key_sets)))
    except ValueError:
        return None


def _all_over_scalar_and_top_level(exprs: Sequence[Expr]) -> bool:
    """
    Return True if every GroupedWindow in exprs is top-level and purely scalar.

    Top-level means the GroupedWindow is the direct value of a NamedExpr (not
    nested inside another expression).  Scalar means all named_aggs are Agg/Len
    reductions with Col partition-by keys.

    Parameters
    ----------
    exprs
        The values of the NamedExprs in a Select.exprs or HStack.columns.

    Returns
    -------
    bool
        True only when it is safe to use the scalar broadcast path.
    """
    for e in exprs:
        if isinstance(e, GroupedWindow):
            _, unary_ops = e._split_named_expr()
            if any(ops for ops in unary_ops.values()):
                return False
            if not all(isinstance(c, Col) for c in e.children[: e.by_count]):
                return False
        elif any(isinstance(node, GroupedWindow) for node in traversal(e.children)):
            return False
    return True


def _contains_unsupported_fill_strategy(exprs: Sequence[Expr]) -> bool:
    for e in traversal(exprs):
        if (
            isinstance(e, UnaryFunction)
            and e.name == "fill_null_with_strategy"
            and e.options[0] not in ("zero", "one")
        ):
            return True
    return False
