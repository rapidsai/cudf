# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel GroupBy Logic."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.dsl.expr import Agg, BinOp, Col, Len, Literal, NamedExpr
from cudf_polars.dsl.ir import GroupBy, Select
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.base import PartitionInfo, _concat, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


# Supported multi-partition aggregations
_GB_AGG_SUPPORTED = ("sum", "count", "mean", "min", "max")


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
    name: str, expr: Expr, *, names: Generator[str, None, None]
) -> tuple[NamedExpr, list[NamedExpr], list[NamedExpr]]:
    """
    Decompose a groupby-aggregation expression.

    Parameters
    ----------
    name
        Output schema name.
    expr
        The aggregation expression for a single column.
    names
        Generator of unique names for temporaries.

    Returns
    -------
    NamedExpr
        The expression selecting the *output* column or columns.
    list[NamedExpr]
        The initial aggregation expressions.
    list[NamedExpr]
        The reduction expressions.
    """
    dtype = expr.dtype

    if isinstance(expr, Len):
        selection = NamedExpr(name, Col(dtype, name))
        aggregation = [NamedExpr(name, expr)]
        reduction = [NamedExpr(name, Agg(dtype, "sum", None, Col(dtype, name)))]
        return selection, aggregation, reduction
    if isinstance(expr, Agg):
        if expr.name in ("sum", "count", "min", "max"):
            if expr.name in ("sum", "count"):
                aggfunc = "sum"
            else:
                aggfunc = expr.name
            selection = NamedExpr(name, Col(dtype, name))
            aggregation = [NamedExpr(name, expr)]
            reduction = [NamedExpr(name, Agg(dtype, aggfunc, None, Col(dtype, name)))]
            return selection, aggregation, reduction
        elif expr.name == "mean":
            (child,) = expr.children
            (sum, count), aggregations, reductions = combine(
                decompose(
                    f"{next(names)}__mean_sum",
                    Agg(dtype, "sum", None, child),
                    names=names,
                ),
                decompose(f"{next(names)}__mean_count", Len(dtype), names=names),
            )
            selection = NamedExpr(
                name,
                BinOp(dtype, plc.binaryop.BinaryOperator.DIV, sum.value, count.value),
            )
            return selection, aggregations, reductions
        else:
            raise NotImplementedError(
                "GroupBy does not support multiple partitions "
                f"for this aggregation type:\n{type(expr)}\n"
                f"Only {_GB_AGG_SUPPORTED} are supported."
            )
    elif isinstance(expr, Literal):
        selection = NamedExpr(name, Col(dtype, name))
        aggregation = []
        reduction = [NamedExpr(name, expr)]
        return selection, aggregation, reduction
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

    temporary_names = unique_names("_" * max(map(len, ir.schema)))
    # Decompose the aggregation requests into three distinct phases
    selection_exprs, piecewise_exprs, reduction_exprs = combine(
        *(
            decompose(agg.name, agg.value, names=temporary_names)
            for agg in ir.agg_requests
        )
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
        None,
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

        gb_inter = Shuffle(
            pwise_schema,
            ir.keys,
            ir.config_options,
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
        ir.zlice,
        ir.config_options,
        gb_inter,
    )
    partition_info[gb_reduce] = PartitionInfo(count=post_aggregation_count)

    # Final Select phase
    new_node = Select(
        ir.schema,
        [
            *(NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in ir.keys),
            *selection_exprs,
        ],
        False,  # noqa: FBT003
        gb_reduce,
    )
    partition_info[new_node] = PartitionInfo(count=post_aggregation_count)
    return new_node, partition_info


def _tree_node(
    do_evaluate: Callable[..., DataFrame], nbatch: int, *args: DataFrame
) -> DataFrame:
    return do_evaluate(*args[nbatch:], _concat(*args[:nbatch]))


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
                len(batch),
                *batch,
                *ir._non_child_args,
            )
            new_keys.append((name, j, i))
        j += 1
        keys = new_keys
    graph[(name, 0)] = (
        _tree_node,
        ir.do_evaluate,
        len(keys),
        *keys,
        *ir._non_child_args,
    )
    return graph
