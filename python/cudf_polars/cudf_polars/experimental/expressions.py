# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-partition Expr classes and utilities.

This module includes the necessary functionality to
decompose a non-pointwise expression graph into stages
that can each be mapped onto a simple partition-wise
task graph at execution time.

For example, if ``Select.exprs`` contains an ``expr.Agg``
node, ``decompose_expr_graph`` will decompose the complex
NamedExpr node into a sequence of three new IR nodes::

 - Select: Partition-wise aggregation logic.
 - Repartition: Concatenate the results of each partition.
 - Select: Final aggregation on the combined results.

In this example, the Select stages are mapped onto a simple
partition-wise task graph at execution time, and the Repartition
stage is used to capture the data-movement required for a global
aggregation. At the moment, data movement is always introduced
by either repartitioning or shuffling the data.

Since we are introducing intermediate IR nodes, we are also
introducing a temporary column for each intermediate result.
In order to avoid column-name collisions with the original
input-IR node, we generate unique names for temporary columns
and concatenate them to the input-IR node using ``HConcat``.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl.expressions.aggregation import Agg
from cudf_polars.dsl.expressions.base import Col, Expr, NamedExpr
from cudf_polars.dsl.expressions.binaryop import BinOp
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.dsl.expressions.unary import Cast, UnaryFunction
from cudf_polars.dsl.ir import Distinct, Empty, HConcat, Select
from cudf_polars.dsl.traversal import (
    CachingVisitor,
)
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.utils import _leaf_column_names

if TYPE_CHECKING:
    from collections.abc import Generator, MutableMapping, Sequence
    from typing import TypeAlias

    from cudf_polars.dsl.expressions.base import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.typing import GenericTransformer, Schema
    from cudf_polars.utils.config import ConfigOptions


ExprDecomposer: TypeAlias = (
    "GenericTransformer[Expr, tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]]"
)


def select(
    exprs: Sequence[Expr],
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    *,
    names: Generator[str, None, None],
    repartition: bool = False,
) -> tuple[list[Col], IR, MutableMapping[IR, PartitionInfo]]:
    """
    Select expressions from an IR node, introducing temporaries.

    Parameters
    ----------
    exprs
        Expressions to select.
    input_ir
        The input IR node to select from.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    names
        Generator of unique names for temporaries.
    repartition
        Whether to add a Repartition node after the
        new selection.

    Returns
    -------
    columns
        Expressions to select from the new IR output.
    new_ir
        The new IR node that will introduce temporaries.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    output_names = [next(names) for _ in range(len(exprs))]
    named_exprs = [
        NamedExpr(name, expr) for name, expr in zip(output_names, exprs, strict=True)
    ]
    new_ir: IR = Select(
        {ne.name: ne.value.dtype for ne in named_exprs},
        named_exprs,
        True,  # noqa: FBT003
        input_ir,
    )
    partition_info[new_ir] = PartitionInfo(count=partition_info[input_ir].count)

    # Optionally collapse into one output partition
    if repartition:
        new_ir = Repartition(new_ir.schema, new_ir)
        partition_info[new_ir] = PartitionInfo(count=1)

    columns = [Col(ne.value.dtype, ne.name) for ne in named_exprs]
    return columns, new_ir, partition_info


def _decompose_unique(
    unique: UnaryFunction,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    *,
    names: Generator[str, None, None],
) -> tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose a 'unique' UnaryFunction into partition-wise stages.

    Parameters
    ----------
    unique
        The expression node to decompose.
    input_ir
        The original input-IR node that ``unique`` will evaluate.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.
    names
        Generator of unique names for temporaries.

    Returns
    -------
    expr
        Decomposed expression node.
    input_ir
        The rewritten ``input_ir`` to be evaluated by ``expr``.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    from cudf_polars.experimental.distinct import lower_distinct

    (child,) = unique.children
    (maintain_order,) = unique.options
    columns, input_ir, partition_info = select(
        [child],
        input_ir,
        partition_info,
        names=names,
    )
    (column,) = columns

    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in '_decompose_unique'"
    )

    cardinality: float | None = None
    if cardinality_factor := {
        max(min(v, 1.0), 0.00001)
        for k, v in config_options.executor.cardinality_factor.items()
        if k in _leaf_column_names(child)
    }:
        cardinality = max(cardinality_factor)

    input_ir, partition_info = lower_distinct(
        Distinct(
            {column.name: column.dtype},
            plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
            None,
            None,
            maintain_order,
            input_ir,
        ),
        input_ir,
        partition_info,
        config_options,
        cardinality=cardinality,
    )

    return column, input_ir, partition_info


def _decompose_agg_node(
    agg: Agg,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    *,
    names: Generator[str, None, None],
) -> tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose an agg expression into partition-wise stages.

    Parameters
    ----------
    agg
        The Agg node to decompose.
    input_ir
        The original input-IR node that ``agg`` will evaluate.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.
    names
        Generator of unique names for temporaries.

    Returns
    -------
    expr
        Decomposed Agg node.
    input_ir
        The rewritten ``input_ir`` to be evaluated by ``expr``.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    expr: Expr
    exprs: list[Expr]
    if agg.name == "count":
        # Chunkwise stage
        columns, input_ir, partition_info = select(
            [agg],
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )

        # Combined stage
        (column,) = columns
        columns, input_ir, partition_info = select(
            [Agg(agg.dtype, "sum", None, column)],
            input_ir,
            partition_info,
            names=names,
        )
        (expr,) = columns
    elif agg.name == "mean":
        # Chunkwise stage
        exprs = [
            Agg(agg.dtype, "sum", None, *agg.children),
            Agg(agg.dtype, "count", None, *agg.children),
        ]
        columns, input_ir, partition_info = select(
            exprs,
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )

        # Combined stage
        exprs = [
            BinOp(
                agg.dtype,
                plc.binaryop.BinaryOperator.DIV,
                *(Agg(agg.dtype, "sum", None, column) for column in columns),
            )
        ]
        columns, input_ir, partition_info = select(
            exprs,
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )
        (expr,) = columns
    elif agg.name == "n_unique":
        # Get uniques and shuffle (if necessary)
        # TODO: Should this be a tree reduction by default?
        (child,) = agg.children
        pi = partition_info[input_ir]
        if pi.count > 1 and [ne.value for ne in pi.partitioned_on] != [input_ir]:
            from cudf_polars.experimental.shuffle import Shuffle

            children, input_ir, partition_info = select(
                [UnaryFunction(agg.dtype, "unique", (False,), child)],
                input_ir,
                partition_info,
                names=names,
            )
            (child,) = children
            agg = agg.reconstruct([child])
            shuffle_on = (NamedExpr(next(names), child),)

            assert config_options.executor.name == "streaming", (
                "'in-memory' executor not supported in '_decompose_agg_node'"
            )

            input_ir = Shuffle(
                input_ir.schema,
                shuffle_on,
                config_options.executor.shuffle_method,
                input_ir,
            )
            partition_info[input_ir] = PartitionInfo(
                count=pi.count,
                partitioned_on=shuffle_on,
            )

        # Chunkwise stage
        columns, input_ir, partition_info = select(
            [Cast(agg.dtype, agg)],
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )

        # Combined stage
        (column,) = columns
        columns, input_ir, partition_info = select(
            [Agg(agg.dtype, "sum", None, column)],
            input_ir,
            partition_info,
            names=names,
        )
        (expr,) = columns
    else:
        # Chunkwise stage
        columns, input_ir, partition_info = select(
            [agg],
            input_ir,
            partition_info,
            names=names,
            repartition=True,
        )

        # Combined stage
        (column,) = columns
        columns, input_ir, partition_info = select(
            [Agg(agg.dtype, agg.name, agg.options, column)],
            input_ir,
            partition_info,
            names=names,
        )
        (expr,) = columns

    return expr, input_ir, partition_info


_SUPPORTED_AGGS = ("count", "min", "max", "sum", "mean", "n_unique")


def _decompose_expr_node(
    expr: Expr,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    *,
    names: Generator[str, None, None],
) -> tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose an expression into partition-wise stages.

    Parameters
    ----------
    expr
        The Expr node to decompose.
    input_ir
        The input IR node that ``expr`` will evaluate.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.
    names
        Generator of unique names for temporaries.

    Returns
    -------
    expr
        Decomposed Expr node.
    input_ir
        The rewritten ``input_ir`` to be evaluated by ``expr``.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    if isinstance(expr, Literal):
        # For Literal nodes, we don't actually want an
        # input IR with real columns, because it will
        # mess up the result of ``HConcat``.
        input_ir = Empty()
        partition_info[input_ir] = PartitionInfo(count=1)

    partition_count = partition_info[input_ir].count
    if partition_count == 1 or expr.is_pointwise:
        # Single-partition and pointwise expressions are always supported.
        return expr, input_ir, partition_info
    elif isinstance(expr, Agg) and expr.name in _SUPPORTED_AGGS:
        # This is a supported Agg expression.
        return _decompose_agg_node(
            expr, input_ir, partition_info, config_options, names=names
        )
    elif isinstance(expr, UnaryFunction) and expr.name == "unique":
        return _decompose_unique(
            expr, input_ir, partition_info, config_options, names=names
        )
    else:
        # This is an un-supported expression - raise.
        raise NotImplementedError(
            f"{type(expr)} not supported for multiple partitions."
        )


def _decompose(
    expr: Expr, rec: ExprDecomposer
) -> tuple[Expr, IR, MutableMapping[IR, PartitionInfo]]:
    # Used by `decompose_expr_graph``

    if not expr.children:
        # Leaf node
        return _decompose_expr_node(
            expr,
            rec.state["input_ir"],
            {rec.state["input_ir"]: rec.state["input_partition_info"]},
            rec.state["config_options"],
            names=rec.state["unique_names"],
        )

    # Process child Exprs first
    children, input_irs, _partition_info = zip(
        *(rec(c) for c in expr.children), strict=True
    )
    partition_info = reduce(operator.or_, _partition_info)

    # Assume the partition count is the maximum input-IR partition count
    input_ir: IR
    assert len(input_irs) > 0  # Must have at least one input IR
    partition_count = max(partition_info[ir].count for ir in input_irs)
    unique_input_irs = [k for k in dict.fromkeys(input_irs) if not isinstance(k, Empty)]
    if len(unique_input_irs) > 1:
        # Need to make sure we only have a single input IR
        # TODO: Check that we aren't concatenating misaligned
        # columns that cannot be broadcasted. For example, what
        # if one of the columns is sorted?
        schema: Schema = {}
        for ir in unique_input_irs:
            schema.update(ir.schema)
        input_ir = HConcat(
            schema,
            True,  # noqa: FBT003
            *unique_input_irs,
        )
        partition_info[input_ir] = PartitionInfo(count=partition_count)
    else:
        input_ir = unique_input_irs[0]

    # Call into class-specific logic to decompose ``expr``
    return _decompose_expr_node(
        expr.reconstruct(children),
        input_ir,
        partition_info,
        rec.state["config_options"],
        names=rec.state["unique_names"],
    )


def decompose_expr_graph(
    named_expr: NamedExpr,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
) -> tuple[NamedExpr, IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose a NamedExpr into stages.

    Parameters
    ----------
    named_expr
        The original NamedExpr to decompose.
    input_ir
        The input-IR node that ``named_expr`` will be
        evaluated on.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    named_expr
        Decomposed NamedExpr object.
    input_ir
        The rewritten ``input_ir`` to be evaluated by ``named_expr``.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.

    Notes
    -----
    This function recursively decomposes ``named_expr.value`` and
    ``input_ir`` into multiple partition-wise stages.
    """
    state = {
        "input_ir": input_ir,
        "input_partition_info": partition_info[input_ir],
        "config_options": config_options,
        "unique_names": unique_names((named_expr.name, *input_ir.schema.keys())),
    }
    mapper = CachingVisitor(_decompose, state=state)
    expr, input_ir, partition_info = mapper(named_expr.value)
    return named_expr.reconstruct(expr), input_ir, partition_info
