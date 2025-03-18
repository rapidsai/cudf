# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Select Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import Select
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import PartitionInfo, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.experimental.expressions import (
    FusedExpr,
    decompose_expr_graph,
    extract_partition_counts,
    make_fusedexpr_graph,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.expressions.base import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


def decompose_select(
    ir: Select,
    child: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Decompose a Select expression (if possible)."""
    named_exprs = [e.reconstruct(decompose_expr_graph(e.value)) for e in ir.exprs]
    new_node = Select(
        ir.schema,
        named_exprs,
        ir.should_broadcast,
        child,
    )
    expr_partition_info = extract_partition_counts(
        named_exprs, partition_info[child].count
    )
    partition_info[new_node] = PartitionInfo(
        count=max(expr_partition_info[ne.value] for ne in named_exprs)
    )
    return new_node, partition_info


@lower_ir_node.register(Select)
def _(
    ir: Select, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])
    pi = partition_info[child]
    if pi.count > 1 and not all(
        expr.is_pointwise for expr in traversal([e.value for e in ir.exprs])
    ):
        # Try decomposing the underlying expressions
        return decompose_select(ir, child, partition_info)

    new_node = ir.reconstruct([child])
    partition_info[new_node] = pi
    return new_node, partition_info


def build_fusedexpr_select_graph(
    ir: Select, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    """Build complex Select graph."""
    (child,) = ir.children
    child_count = partition_info[child].count

    # Build Graph to produce a column for each
    # NamedExpr element of ir.exprs
    graph: MutableMapping[Any, Any] = {}
    expr_partition_counts: MutableMapping[Expr, int] = {}
    roots = []
    for ne in ir.exprs:
        assert isinstance(ne.value, FusedExpr), f"{ne.value} is not a FusedExpr"
        roots.append(ne)
        expr_partition_counts = extract_partition_counts(
            [ne],
            child_count,
            update=expr_partition_counts,
        )
        for node in traversal([ne.value]):
            assert isinstance(node, FusedExpr), f"{node} is not a FusedExpr"
            graph.update(
                make_fusedexpr_graph(
                    ne.reconstruct(node),
                    expr_partition_counts,
                    child,
                    partition_info[child],
                )
            )

    # Add task(s) to select the final columns
    name = get_key_name(ir)
    count = max(expr_partition_counts[root.value] for root in roots)
    expr_names = [get_key_name(root.value) for root in roots]
    expr_bcast = [expr_partition_counts[root.value] == 1 for root in roots]
    for i in range(count):
        graph[(name, i)] = (
            DataFrame,
            [
                (name, 0) if bcast else (name, i)
                for name, bcast in zip(expr_names, expr_bcast, strict=True)
            ],
        )

    return graph


@generate_ir_tasks.register(Select)
def _(
    ir: Select, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    # TODO: If we are doing aligned aggregations on multiple
    # columns at once, we should build a task graph that
    # evaluates the aligned expressions at the same time
    # (rather than each task operating on an individual column).

    fused_exprs = [isinstance(ne.value, FusedExpr) for ne in ir.exprs]
    if any(fused_exprs):
        # Handle FusedExpr-based graph construction
        assert all(fused_exprs), "Partial fusion is not supported"
        return build_fusedexpr_select_graph(ir, partition_info)
    else:
        # Simple point-wise graph
        child_name = get_key_name(ir.children[0])
        return {
            key: (
                ir.do_evaluate,
                *ir._non_child_args,
                (child_name, i),
            )
            for i, key in enumerate(partition_info[ir].keys(ir))
        }
