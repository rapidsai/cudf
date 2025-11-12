# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel GroupBy Logic."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl.expr import Agg, BinOp, Col, Len, NamedExpr
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.ir import GroupBy, Select, Slice
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.shuffle import Shuffle
from cudf_polars.experimental.utils import _get_unique_fractions, _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import Generator, MutableMapping

    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


# Supported multi-partition aggregations
_GB_AGG_SUPPORTED = ("sum", "count", "mean", "min", "max", "n_unique")


def combine(
    *decompositions: tuple[NamedExpr, list[NamedExpr], list[NamedExpr], bool],
) -> tuple[list[NamedExpr], list[NamedExpr], list[NamedExpr], bool]:
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
    if len(decompositions) == 0:
        return [], [], [], False
    selections, aggregations, reductions, need_preshuffles = zip(
        *decompositions, strict=True
    )
    assert all(isinstance(ne, NamedExpr) for ne in selections)
    return (
        list(selections),
        list(itertools.chain.from_iterable(aggregations)),
        list(itertools.chain.from_iterable(reductions)),
        any(need_preshuffles),
    )


def decompose(
    name: str, expr: Expr, *, names: Generator[str, None, None]
) -> tuple[NamedExpr, list[NamedExpr], list[NamedExpr], bool]:
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
    bool
        Whether we need to pre-shuffle on the group_by keys.
    """
    dtype = expr.dtype

    if isinstance(expr, Len):
        selection = NamedExpr(name, Col(dtype, name))
        aggregation = [NamedExpr(name, expr)]
        reduction = [
            NamedExpr(
                name,
                Agg(dtype, "sum", None, ExecutionContext.GROUPBY, Col(dtype, name)),
            )
        ]
        return selection, aggregation, reduction, False
    if isinstance(expr, Agg):
        if expr.name in ("sum", "count", "min", "max", "n_unique"):
            if expr.name in ("sum", "count", "n_unique"):
                aggfunc = "sum"
            else:
                aggfunc = expr.name
            selection = NamedExpr(name, Col(dtype, name))
            aggregation = [NamedExpr(name, expr)]
            reduction = [
                NamedExpr(
                    name,
                    Agg(
                        dtype, aggfunc, None, ExecutionContext.GROUPBY, Col(dtype, name)
                    ),
                )
            ]
            return selection, aggregation, reduction, expr.name == "n_unique"
        elif expr.name == "mean":
            (child,) = expr.children
            (sum, count), aggregations, reductions, need_preshuffle = combine(
                decompose(
                    f"{next(names)}__mean_sum",
                    Agg(dtype, "sum", None, ExecutionContext.GROUPBY, child),
                    names=names,
                ),
                decompose(
                    f"{next(names)}__mean_count",
                    Agg(
                        DataType(pl.Int32()),
                        "count",
                        False,  # noqa: FBT003
                        ExecutionContext.GROUPBY,
                        child,
                    ),
                    names=names,
                ),
            )
            selection = NamedExpr(
                name,
                BinOp(dtype, plc.binaryop.BinaryOperator.DIV, sum.value, count.value),
            )
            return selection, aggregations, reductions, need_preshuffle
        else:
            raise NotImplementedError(
                "group_by does not support multiple partitions "
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
    # Pull slice operations out of the GroupBy before lowering
    if ir.zlice is not None:
        offset, length = ir.zlice
        if length is None:  # pragma: no cover
            return _lower_ir_fallback(
                ir,
                rec,
                msg="This slice not supported for multiple partitions.",
            )
        new_join = GroupBy(
            ir.schema,
            ir.keys,
            ir.agg_requests,
            ir.maintain_order,
            None,
            *ir.children,
        )
        return rec(Slice(ir.schema, offset, length, new_join))

    # Extract child partitioning
    original_child = ir.children[0]
    child, partition_info = rec(ir.children[0])

    # Handle single-partition case
    if partition_info[child].count == 1:
        single_part_node = ir.reconstruct([child])
        partition_info[single_part_node] = partition_info[child]
        return single_part_node, partition_info

    # Check group-by keys
    if not all(
        expr.is_pointwise for expr in traversal([e.value for e in ir.keys])
    ):  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg="group_by does not support multiple partitions for non-pointwise keys.",
        )

    # Check if we are dealing with any high-cardinality columns
    post_aggregation_count = 1  # Default tree reduction
    groupby_key_columns = [ne.name for ne in ir.keys]
    shuffled = partition_info[child].partitioned_on == ir.keys

    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node'"
    )

    child_count = partition_info[child].count
    if unique_fraction_dict := _get_unique_fractions(
        groupby_key_columns,
        config_options.executor.unique_fraction,
        row_count=rec.state["stats"].row_count.get(original_child),
        column_stats=rec.state["stats"].column_stats.get(original_child),
    ):
        # Use unique_fraction to determine output partitioning
        unique_fraction = max(unique_fraction_dict.values())
        post_aggregation_count = max(int(unique_fraction * child_count), 1)

    new_node: IR
    name_generator = unique_names(ir.schema.keys())
    # Decompose the aggregation requests into three distinct phases
    try:
        selection_exprs, piecewise_exprs, reduction_exprs, need_preshuffle = combine(
            *(
                decompose(agg.name, agg.value, names=name_generator)
                for agg in ir.agg_requests
            )
        )
    except NotImplementedError:
        if shuffled:  # pragma: no cover
            # Don't fallback if we are already shuffled.
            # We can just preserve the child's partitioning
            new_node = ir.reconstruct([child])
            partition_info[new_node] = partition_info[child]
            return new_node, partition_info
        return _lower_ir_fallback(
            ir, rec, msg="Failed to decompose groupby aggs for multiple partitions."
        )

    # Preshuffle ir.child if needed
    if need_preshuffle:
        child = Shuffle(
            child.schema,
            ir.keys,
            config_options.executor.shuffle_method,
            child,
        )
        partition_info[child] = PartitionInfo(
            count=child_count,
            partitioned_on=ir.keys,
        )
        shuffled = True

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
        child,
    )
    child_count = partition_info[child].count
    partition_info[gb_pwise] = PartitionInfo(count=child_count)
    grouped_keys = tuple(NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in ir.keys)

    # Reduction
    gb_inter: GroupBy | Repartition | Shuffle
    reduction_schema = {k.name: k.value.dtype for k in grouped_keys} | {
        k.name: k.value.dtype for k in reduction_exprs
    }
    if not shuffled and post_aggregation_count > 1:
        # Shuffle reduction
        if ir.maintain_order:  # pragma: no cover
            return _lower_ir_fallback(
                ir,
                rec,
                msg="maintain_order not supported for multiple output partitions.",
            )

        gb_inter = Shuffle(
            gb_pwise.schema,
            grouped_keys,
            config_options.executor.shuffle_method,
            gb_pwise,
        )
        partition_info[gb_inter] = PartitionInfo(count=post_aggregation_count)
    else:
        # N-ary tree reduction
        assert config_options.executor.name == "streaming", (
            "'in-memory' executor not supported in 'generate_ir_tasks'"
        )

        n_ary = config_options.executor.groupby_n_ary
        count = child_count
        gb_inter = gb_pwise
        while count > post_aggregation_count:
            gb_inter = Repartition(gb_inter.schema, gb_inter)
            count = max(math.ceil(count / n_ary), post_aggregation_count)
            partition_info[gb_inter] = PartitionInfo(count=count)
            if count > post_aggregation_count:
                gb_inter = GroupBy(
                    reduction_schema,
                    grouped_keys,
                    reduction_exprs,
                    ir.maintain_order,
                    None,
                    gb_inter,
                )
                partition_info[gb_inter] = PartitionInfo(count=count)

    # Final aggregation
    gb_reduce = GroupBy(
        reduction_schema,
        grouped_keys,
        reduction_exprs,
        ir.maintain_order,
        ir.zlice,
        gb_inter,
    )
    partition_info[gb_reduce] = PartitionInfo(count=post_aggregation_count)

    # Final Select phase
    new_node = Select(
        ir.schema,
        [
            *(NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in grouped_keys),
            *selection_exprs,
        ],
        False,  # noqa: FBT003
        gb_reduce,
    )
    partition_info[new_node] = PartitionInfo(
        count=post_aggregation_count,
        partitioned_on=grouped_keys,
    )
    return new_node, partition_info
