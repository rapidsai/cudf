# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition evaluation."""

from __future__ import annotations

import operator
from functools import partial, reduce
from typing import TYPE_CHECKING

import polars as pl

import cudf_polars.experimental.distinct
import cudf_polars.experimental.groupby
import cudf_polars.experimental.io
import cudf_polars.experimental.join
import cudf_polars.experimental.select
import cudf_polars.experimental.shuffle
import cudf_polars.experimental.sort  # noqa: F401
from cudf_polars.containers import DataType
from cudf_polars.dsl.expr import Col, Literal, NamedExpr
from cudf_polars.dsl.ir import (
    IR,
    Cache,
    Filter,
    HConcat,
    HStack,
    MapFunction,
    Projection,
    Select,
    Slice,
    Union,
)
from cudf_polars.dsl.traversal import CachingVisitor, traversal
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.io import _clear_source_info_cache
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.utils import (
    _contains_over,
    _dynamic_planning_on,
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.experimental.base import StatsCollector
    from cudf_polars.experimental.dispatch import LowerIRTransformer, State
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


@lower_ir_node.register(IR)
def _(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:  # pragma: no cover
    # Default logic - Requires single partition
    return _lower_ir_fallback(
        ir, rec, msg=f"Class {type(ir)} does not support multiple partitions."
    )


def lower_ir_graph(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR graph and extract partitioning information.

    Parameters
    ----------
    ir
        Root of the graph to rewrite.
    config_options
        GPUEngine configuration options.
    stats
        Pre-computed statistics collector.

    Returns
    -------
    new_ir, partition_info
        The rewritten graph and a mapping from unique nodes
        in the new graph to associated partitioning information.

    Notes
    -----
    This function traverses the unique nodes of the graph with
    root `ir`, and applies :func:`lower_ir_node` to each node.

    See Also
    --------
    lower_ir_node
    """
    state: State = {
        "config_options": config_options,
        "stats": stats,
    }
    mapper: LowerIRTransformer = CachingVisitor(lower_ir_node, state=state)
    return mapper(ir)


def evaluate_rapidsmpf(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
) -> pl.DataFrame:  # pragma: no cover; rapidsmpf runtime not tested in CI yet
    """
    Evaluate with the RapidsMPF streaming runtime.

    Parameters
    ----------
    ir
        Logical plan to evaluate.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A cudf-polars DataFrame object.
    """
    from cudf_polars.experimental.rapidsmpf.core import evaluate_logical_plan

    result, _ = evaluate_logical_plan(ir, config_options, collect_metadata=False)
    return result


def evaluate_streaming(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
) -> pl.DataFrame:
    """
    Evaluate an IR graph with partitioning.

    Parameters
    ----------
    ir
        Logical plan to evaluate.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A cudf-polars DataFrame object.
    """
    # Clear source info cache in case data was overwritten
    _clear_source_info_cache()

    return evaluate_rapidsmpf(ir, config_options)


@lower_ir_node.register(Union)
def _(
    ir: Union, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Check zlice
    if ir.zlice is not None:
        return rec(
            Slice(
                ir.schema,
                *ir.zlice,
                Union(ir.schema, None, *ir.children),
            )
        )

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Partition count is the sum of all child partitions
    count = sum(partition_info[c].count for c in children)

    # Return reconstructed node and partition-info dict
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=count)
    return new_node, partition_info


@lower_ir_node.register(MapFunction)
def _(
    ir: MapFunction, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Allow pointwise operations
    if ir.name in ("rename", "explode"):
        return _lower_ir_pwise(ir, rec)

    # Fallback for everything else
    return _lower_ir_fallback(
        ir, rec, msg=f"{ir.name} is not supported for multiple partitions."
    )


def _lower_ir_pwise(
    ir: IR, rec: LowerIRTransformer, *, preserve_partitioning: bool = False
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Lower a partition-wise (i.e. embarrassingly-parallel) IR node

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)
    counts = {partition_info[c].count for c in children}

    # Check that child partitioning is supported
    if len(counts) > 1:  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg=f"Class {type(ir)} does not support children with mismatched partition counts.",
        )

    # Preserve child partition_info if possible
    if preserve_partitioning and len(children) == 1:
        partition = partition_info[children[0]]
    else:
        partition = PartitionInfo(count=max(counts))

    # Return reconstructed node and partition-info dict
    new_node = ir.reconstruct(children)
    partition_info[new_node] = partition
    return new_node, partition_info


_lower_ir_pwise_preserve = partial(_lower_ir_pwise, preserve_partitioning=True)
lower_ir_node.register(Cache, _lower_ir_pwise_preserve)
lower_ir_node.register(HConcat, _lower_ir_pwise)


@lower_ir_node.register(Filter)
def _(
    ir: Filter, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])

    if partition_info[child].count > 1 and _contains_over([ir.mask.value]):
        # mask contains .over(...), collapse to single partition
        return _lower_ir_fallback(
            ir.reconstruct([child]),
            rec,
            msg=(
                "over(...) inside filter is not supported for multiple partitions; "
                "falling back to in-memory evaluation."
            ),
        )

    if partition_info[child].count > 1 and not all(
        expr.is_pointwise for expr in traversal([ir.mask.value])
    ):
        # TODO: Use expression decomposition to lower Filter
        # See: https://github.com/rapidsai/cudf/issues/20076
        return _lower_ir_fallback(
            ir, rec, msg="This filter is not supported for multiple partitions."
        )

    new_node = ir.reconstruct([child])
    partition_info[new_node] = partition_info[child]
    return new_node, partition_info


@lower_ir_node.register(Slice)
def _(
    ir: Slice, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Check for dynamic planning - may have more partitions at runtime
    config_options = rec.state["config_options"]
    dynamic_planning = _dynamic_planning_on(config_options)

    if ir.offset == 0:
        # Taking the first N rows.
        # We don't know how large each partition is, so we reduce.
        new_node, partition_info = _lower_ir_pwise(ir, rec)
        if partition_info[new_node].count > 1 or dynamic_planning:
            # Collapse down to single partition
            inter = Repartition(new_node.schema, new_node)
            partition_info[inter] = PartitionInfo(count=1)
            # Slice reduced partition
            new_node = ir.reconstruct([inter])
            partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info

    # Fallback
    return _lower_ir_fallback(
        ir, rec, msg="This slice not supported for multiple partitions."
    )


def _add_anchor_column(ir: HStack) -> tuple[HStack, str, DataType]:
    """Add temporary anchor column to preserve row count."""
    anchor_name = next(unique_names((*ir.schema, *ir.children[0].schema)))
    anchor_dtype = DataType(pl.datatypes.Int8())
    anchor_named_expr = NamedExpr(anchor_name, Literal(anchor_dtype, 0))
    new_ir = HStack(
        ir.children[0].schema | {anchor_name: anchor_dtype},
        (anchor_named_expr,),
        True,  # noqa: FBT003
        ir.children[0],
    )
    return new_ir, anchor_name, anchor_dtype


@lower_ir_node.register(HStack)
def _(
    ir: HStack, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if not all(e.is_pointwise for e in traversal([ne.value for ne in ir.columns])):
        # Redirect non-pointwise HStack to Select so the Select handler can
        # attempt decomposition (or fall back gracefully via decompose_select).
        child: IR = ir.children[0]
        anchor_name: str | None = None
        col_map = {ne.name: ne for ne in ir.columns}
        schema = ir.schema
        if ir.should_broadcast and all(name in col_map for name in ir.schema):
            # We need to add a temporary anchor column to preserve row count.
            child, anchor_name, anchor_dtype = _add_anchor_column(ir)

            schema = ir.schema | {anchor_name: anchor_dtype}
        exprs = tuple(
            col_map[name] if name in col_map else NamedExpr(name, Col(dtype, name))
            for name, dtype in schema.items()
        )
        new_ir: Select | Projection = Select(schema, exprs, ir.should_broadcast, child)
        if anchor_name is not None:
            # Need to drop the temporary anchor column.
            schema = {
                name: dtype
                for name, dtype in new_ir.schema.items()
                if name != anchor_name
            }
            new_ir = Projection(schema, new_ir)
        return lower_ir_node(new_ir, rec)

    child, partition_info = rec(ir.children[0])
    new_node = ir.reconstruct([child])
    partition_info[new_node] = partition_info[child]
    return new_node, partition_info
