# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel GroupBy Logic."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.dsl.expr import Agg, BinOp, Cast, Col, Len, NamedExpr, UnaryFunction
from cudf_polars.dsl.ir import GroupBy, Select
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import PartitionInfo, _concat, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


# Supported multi-partition aggregations
_GB_AGG_SUPPORTED = ("sum", "count", "mean")


def combine(
    *decompositions: tuple[NamedExpr, list[NamedExpr], list[NamedExpr]],
) -> tuple[list[NamedExpr], list[NamedExpr], list[NamedExpr]]:
    """
    Combine multiple groupby-aggregation decompositions.

    Parameters
    ----------
    decompositions
        Packed sequence of `decompose` results.

    Returns
    -------
    Unified groupby-aggregation decomposition.
    """
    selections, aggregations, reductions = zip(*decompositions, strict=True)
    assert all(isinstance(ne, NamedExpr) for ne in selections)
    return (
        list(selections),
        list(itertools.chain.from_iterable(aggregations)),
        list(itertools.chain.from_iterable(reductions)),
    )


def decompose(
    name: str, expr: Expr
) -> tuple[NamedExpr, list[NamedExpr], list[NamedExpr]]:
    """
    Decompose a groupby-aggregation expression.

    Parameters
    ----------
    name
        Output schema name.
    expr
        The aggregation expression for a single column.

    Returns
    -------
    Tuple containing the `NamedExpr`s for each of the
    three parallel-aggregation phases: (1) selection,
    (2) initial aggregation, and (3) reduction.
    """
    dtype = expr.dtype
    expr = expr.children[0] if isinstance(expr, Cast) else expr

    unary_op: list[Any] = []
    if isinstance(expr, UnaryFunction) and expr.is_pointwise:
        # TODO: Handle multiple/sequential unary ops
        unary_op = [expr.name, expr.options]
        expr = expr.children[0]

    def _wrap_unary(select):
        # Helper function to wrap the final selection
        # in a UnaryFunction (when necessary)
        if unary_op:
            return UnaryFunction(select.dtype, *unary_op, select)
        return select

    if isinstance(expr, Len):
        selection = NamedExpr(name, _wrap_unary(Col(dtype, name)))
        aggregation = [NamedExpr(name, expr)]
        reduction = [
            NamedExpr(
                name,
                # Sum reduction may require casting.
                # Do it for all cases to be safe (for now)
                Cast(dtype, Agg(dtype, "sum", None, Col(dtype, name))),
            )
        ]
        return selection, aggregation, reduction
    if isinstance(expr, Agg):
        if expr.name in ("sum", "count"):
            selection = NamedExpr(name, _wrap_unary(Col(dtype, name)))
            aggregation = [NamedExpr(name, expr)]
            reduction = [
                NamedExpr(
                    name,
                    # Sum reduction may require casting.
                    # Do it for all cases to be safe (for now)
                    Cast(dtype, Agg(dtype, "sum", None, Col(dtype, name))),
                )
            ]
            return selection, aggregation, reduction
        elif expr.name == "mean":
            (child,) = expr.children
            (sum, count), aggregations, reductions = combine(
                # TODO: Avoid possibility of a name collision
                # (even though the likelihood is small)
                decompose(f"{name}__mean_sum", Agg(dtype, "sum", None, child)),
                decompose(f"{name}__mean_count", Len(dtype)),
            )
            selection = NamedExpr(
                name,
                _wrap_unary(
                    BinOp(
                        dtype, plc.binaryop.BinaryOperator.DIV, sum.value, count.value
                    )
                ),
            )
            return selection, aggregations, reductions
        else:
            raise NotImplementedError(
                "GroupBy does not support multiple partitions "
                f"for this aggregation type:\n{type(expr)}\n"
                f"Only {_GB_AGG_SUPPORTED} are supported."
            )
    else:  # pragma: no cover
        # Unsupported expression
        raise NotImplementedError(
            f"GroupBy does not support multiple partitions for this expression:\n{expr}"
        )


@lower_ir_node.register(GroupBy)
def _(
    ir: GroupBy, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Extract child partitioning
    child, partition_info = rec(ir.children[0])

    # Handle single-partition case
    if partition_info[child].count == 1:
        single_part_node = ir.reconstruct([child])
        partition_info[single_part_node] = partition_info[child]
        return single_part_node, partition_info

    # Check group-by keys
    if not all(expr.is_pointwise for expr in traversal([e.value for e in ir.keys])):
        raise NotImplementedError(
            f"GroupBy does not support multiple partitions for keys:\n{ir.keys}"
        )  # pragma: no cover

    # Check if we are dealing with any high-cardinality columns
    post_aggregation_count = 1  # Default tree reduction
    groupby_key_columns = [ne.name for ne in ir.keys]
    cardinality_factor = {
        c: min(f, 1.0)
        for c, f in ir.config_options.get(
            "executor_options.cardinality_factor", default={}
        ).items()
        if c in groupby_key_columns
    }
    if cardinality_factor:
        # The `cardinality_factor` dictionary can be used
        # to specify a mapping between column names and
        # cardinality "factors". Each factor estimates the
        # fractional number of unique values in the column.
        # Each value should be in the range (0, 1].
        child_count = partition_info[child].count
        post_aggregation_count = max(
            int(max(cardinality_factor.values()) * child_count),
            1,
        )

    # Decompose the aggregation requests into three distinct phases
    selection_exprs, piecewise_exprs, reduction_exprs = combine(
        *(decompose(agg.name, agg.value) for agg in ir.agg_requests)
    )

    # Partition-wise groupby operation
    pwise_schema = {k.name: k.value.dtype for k in ir.keys} | {
        k.name: k.value.dtype for k in piecewise_exprs
    }
    gb_pwise = GroupBy(
        pwise_schema,
        ir.keys,
        piecewise_exprs,
        ir.maintain_order,
        ir.options,
        ir.config_options,
        child,
    )
    child_count = partition_info[child].count
    partition_info[gb_pwise] = PartitionInfo(count=child_count)

    # Add Shuffle node if necessary
    gb_inter: GroupBy | Shuffle = gb_pwise
    if post_aggregation_count > 1:
        if ir.maintain_order:  # pragma: no cover
            raise NotImplementedError(
                "maintain_order not supported for multiple output partitions."
            )

        shuffle_options: dict[str, Any] = {}
        gb_inter = Shuffle(
            pwise_schema,
            ir.keys,
            shuffle_options,
            gb_pwise,
        )
        partition_info[gb_inter] = PartitionInfo(count=post_aggregation_count)

    # Tree reduction if post_aggregation_count==1
    # (Otherwise, this is another partition-wise op)
    gb_reduce = GroupBy(
        {k.name: k.value.dtype for k in ir.keys}
        | {k.name: k.value.dtype for k in reduction_exprs},
        ir.keys,
        reduction_exprs,
        ir.maintain_order,
        ir.options,
        ir.config_options,
        gb_inter,
    )
    partition_info[gb_reduce] = PartitionInfo(count=post_aggregation_count)

    # Final Select phase
    aggregated = {ne.name: ne for ne in selection_exprs}
    new_node = Select(
        ir.schema,
        [
            # Select the aggregated data or the original column
            aggregated.get(name, NamedExpr(name, Col(dtype, name)))
            for name, dtype in ir.schema.items()
        ],
        False,  # noqa: FBT003
        gb_reduce,
    )
    partition_info[new_node] = PartitionInfo(count=post_aggregation_count)
    return new_node, partition_info


def _tree_node(do_evaluate, batch, *args):
    return do_evaluate(*args, _concat(batch))


@generate_ir_tasks.register(GroupBy)
def _(
    ir: GroupBy, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    (child,) = ir.children
    child_count = partition_info[child].count
    child_name = get_key_name(child)
    output_count = partition_info[ir].count

    if output_count == child_count:
        return {
            key: (
                ir.do_evaluate,
                *ir._non_child_args,
                (child_name, i),
            )
            for i, key in enumerate(partition_info[ir].keys(ir))
        }
    elif output_count != 1:  # pragma: no cover
        raise ValueError(f"Expected single partition, got {output_count}")

    # Simple N-ary tree reduction
    j = 0
    n_ary = ir.config_options.get("executor_options.groupby_n_ary", default=32)
    graph: MutableMapping[Any, Any] = {}
    name = get_key_name(ir)
    keys: list[Any] = [(child_name, i) for i in range(child_count)]
    while len(keys) > n_ary:
        new_keys: list[Any] = []
        for i, k in enumerate(range(0, len(keys), n_ary)):
            batch = keys[k : k + n_ary]
            graph[(name, j, i)] = (
                _tree_node,
                ir.do_evaluate,
                batch,
                *ir._non_child_args,
            )
            new_keys.append((name, j, i))
        j += 1
        keys = new_keys
    graph[(name, 0)] = (
        _tree_node,
        ir.do_evaluate,
        keys,
        *ir._non_child_args,
    )
    return graph
