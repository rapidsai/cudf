# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Partitioned LogicalPlan nodes."""

from __future__ import annotations

import warnings
from functools import singledispatch
from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.dsl.expr import Agg, Col, NamedExpr
from cudf_polars.dsl.ir import Scan, Select, broadcast
from cudf_polars.dsl.nodebase import PartitionInfo
from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.nodebase import Node


def get_key_name(node: Node | NamedExpr) -> str:
    """Generate the key name for a Node."""
    if isinstance(node, NamedExpr):
        return f"named-{get_key_name(node.value)}"
    return f"{type(node).__name__.lower()}-{hash(node)}"


@singledispatch
def _ir_partition_info_dispatch(node: Node) -> PartitionInfo:
    count = 1
    if node.children:
        count = max(child.parts.count for child in node.children)
    if count > 1:
        warnings.warn(
            f"Multi-partition support is not implemented for {type(node)}. "
            f"Partitions will be concatenated. Expect poor performance.",
            stacklevel=1,
        )
    return PartitionInfo(count=1)


@singledispatch
def _expr_partition_info_dispatch(
    expr: Expr | NamedExpr, child_ir: IR
) -> PartitionInfo:
    # The default partitioning of an Expr node depends
    # on the child Expr nodes or the child IR node it
    # is operating on (for leaf Exprs).
    if isinstance(expr, NamedExpr):
        expr = expr.value
    if expr.children:
        count = max([child.parts.count for child in expr.children])
    else:
        count = child_ir.parts.count
    if count > 1:
        warnings.warn(
            f"Multi-partition support is not implemented for {type(expr)}. "
            f"Partitions will be concatenated. Expect poor performance.",
            stacklevel=1,
        )
    return PartitionInfo(count=1)


def _default_generate_ir_tasks(ir: IR) -> MutableMapping[Any, Any]:
    # Single-partition default behavior
    if ir.parts.count == 1:
        # Start by concatenating multi-partition
        # children (if necessary)
        graph = {}
        child_names = []
        key_name = get_key_name(ir)
        for child in ir.children:
            child_name_in = get_key_name(child)
            if child.parts.count > 1:
                child_names.append("concat-" + child_name_in)
                graph[(child_names[-1], 0)] = (
                    DataFrame.concat,
                    [(child_name_in, i) for i in range(child.parts.count)],
                )
            else:
                child_names.append(child_name_in)

        # Add single-partition task
        graph[(key_name, 0)] = (
            ir.do_evaluate,
            *ir._non_child_args,
            *((child_name, 0) for child_name in child_names),
        )
        return graph

    raise NotImplementedError(f"Cannot generate tasks for {ir}.")


@singledispatch
def generate_ir_tasks(ir: IR) -> MutableMapping[Any, Any]:
    """
    Generate tasks for an IR node.

    An IR node only needs to generate the graph for
    the current IR logic (not including child IRs).
    """
    return _default_generate_ir_tasks(ir)


@singledispatch
def generate_expr_tasks(
    expr: Expr | NamedExpr, child_ir: IR
) -> MutableMapping[Any, Any]:
    """
    Generate tasks for an Expr node.

    An Expr node is responsible for constructing the full
    expression graph recursively (allowing fusion).
    """
    raise NotImplementedError(f"Cannot generate tasks for {expr}.")


def task_graph(ir: IR) -> tuple[MutableMapping[str, Any], str]:
    """Construct a Dask-compatible task graph."""
    # NOTE: It may be necessary to add an optimization
    # pass here to "rewrite" the single-partition IR graph.

    graph = {
        k: v
        for layer in [generate_ir_tasks(n) for n in traversal(ir)]
        for k, v in layer.items()
    }

    # Add task to reduce output partitions
    key_name = get_key_name(ir)
    if ir.parts.count > 1:
        graph[key_name] = (
            DataFrame.concat,
            [(key_name, i) for i in range(ir.parts.count)],
        )
    else:
        graph[key_name] = (key_name, 0)

    return graph, key_name


def evaluate_dask(ir: IR) -> DataFrame:
    """Evaluate an IR graph with Dask."""
    from dask import get

    graph, key = task_graph(ir)
    return get(graph, key)


##
## Scan
##

_SCAN_SUPPORTED = ("parquet",)


@_ir_partition_info_dispatch.register(Scan)
def _(ir: Scan) -> PartitionInfo:
    if ir.typ in _SCAN_SUPPORTED:
        return PartitionInfo(count=len(ir.paths))
    return PartitionInfo(count=1)


@generate_ir_tasks.register(Scan)
def _(ir: Scan) -> MutableMapping[Any, Any]:
    key_name = get_key_name(ir)
    if ir.parts.count == 1:
        return {
            (key_name, 0): (
                ir.do_evaluate,
                *ir._non_child_args,
            )
        }
    else:
        # Only support 1:1 mapping between
        # paths and partitions for now
        assert len(ir.paths) == ir.parts.count
        return {
            (key_name, i): (
                ir.do_evaluate,
                ir.schema,
                ir.typ,
                ir.reader_options,
                [path],
                ir.with_columns,
                ir.skip_rows,
                ir.n_rows,
                ir.row_index,
                ir.predicate,
            )
            for i, path in enumerate(ir.paths)
        }


##
## Select
##


@_ir_partition_info_dispatch.register(Select)
def _(ir: Select) -> PartitionInfo:
    # Partitioning depends on the expression
    df = ir.children[0]
    column_partition_counts = [
        _expr_partition_info_dispatch(expr, df).count for expr in ir.exprs
    ]
    count = max(column_partition_counts)
    if count > 1:
        warnings.warn(
            f"Multi-partition support is not implemented for {type(ir)}. "
            f"Partitions will be concatenated. Expect poor performance.",
            stacklevel=1,
        )
    return PartitionInfo(count=1)


def _select(columns: list[Column], should_broadcast: bool):  # noqa: FBT001
    if should_broadcast:
        columns = broadcast(*columns)
    return DataFrame(columns)


@generate_ir_tasks.register(Select)
def _(ir: Select) -> MutableMapping[Any, Any]:
    try:
        expr_graphs = [generate_expr_tasks(e, ir.children[0]) for e in ir.exprs]
        key_name = get_key_name(ir)
        expr_keys = [get_key_name(e) for e in ir.exprs]
        graph = {
            (key_name, i): (
                _select,
                [(c_key, i) for c_key in expr_keys],
                ir.should_broadcast,
            )
            for i in range(ir.parts.count)
        }
        for expr_graph in expr_graphs:
            graph.update(expr_graph)
    except NotImplementedError as err:
        if ir.parts.count == 1:
            return _default_generate_ir_tasks(ir)
        raise NotImplementedError("Not supported.") from err
    else:
        return graph


##
## NamedExpr
##


@_expr_partition_info_dispatch.register(NamedExpr)
def _(expr: NamedExpr, child_ir: IR) -> PartitionInfo:
    return _expr_partition_info_dispatch(expr.value, child_ir)


def _rename_column(column: Column, name: str):
    return column.rename(name)


@generate_expr_tasks.register(NamedExpr)
def _(expr: NamedExpr, child_ir: IR) -> MutableMapping[Any, Any]:
    graph = generate_expr_tasks(expr.value, child_ir)
    named_expr_key_name = get_key_name(expr)
    expr_key_name = get_key_name(expr.value)
    for i in range(expr.value.parts.count):
        graph[(named_expr_key_name, i)] = (
            _rename_column,
            graph.pop((expr_key_name, i)),
            expr.name,
        )
    return graph


##
## Col
##


@_expr_partition_info_dispatch.register(Col)
def _(expr: Col, child_ir: IR) -> PartitionInfo:
    assert not expr.children
    count = child_ir.parts.count
    return PartitionInfo(count=count)


def _get_col(df: DataFrame, name: str) -> Column:
    return df.column_map[name].rename(None)


@generate_expr_tasks.register(Col)
def _(expr: Col, child_ir: IR) -> MutableMapping[Any, Any]:
    key_name = get_key_name(expr)
    child_name = get_key_name(child_ir)
    return {
        (key_name, i): (_get_col, (child_name, i), expr.name)
        for i in range(child_ir.parts.count)
    }


##
## Agg
##

_AGG_SUPPORTED = ("sum",)


@_expr_partition_info_dispatch.register(Agg)
def _(expr: Agg, child_ir: IR) -> PartitionInfo:
    if expr.children:
        count = max([child.parts.count for child in expr.children])
    else:
        count = child_ir.parts.count
    if count > 1 and expr.name not in _AGG_SUPPORTED:
        # Only support sum reductions for now.
        warnings.warn(
            f"Multi-partition support is not implemented for {type(expr)}. "
            f"Partitions will be concatenated. Expect poor performance.",
            stacklevel=1,
        )
    return PartitionInfo(count=1)


def _agg_chunk(
    column: Column, request: plc.aggregation.Aggregation, dtype: plc.DataType
) -> Column:
    # TODO: This logic should be different than `request` in many cases
    return Column(
        plc.Column.from_scalar(
            plc.reduce.reduce(column.obj, request, dtype),
            1,
        )
    )


def _agg_combine(columns: Sequence[Column]) -> Column:
    return Column(plc.concatenate.concatenate([col.obj for col in columns]))


def _agg_finalize(
    column: Column,
    request: plc.aggregation.Aggregation,
    dtype: plc.DataType,
) -> Column:
    # TODO: This logic should be different than `request` in many cases
    return Column(
        plc.Column.from_scalar(
            plc.reduce.reduce(column.obj, request, dtype),
            1,
        )
    )


@generate_expr_tasks.register(Agg)
def _(expr: Agg, child_ir: IR) -> MutableMapping[Any, Any]:
    if expr.name not in _AGG_SUPPORTED:
        raise NotImplementedError(f"Cannot generate tasks for {expr}.")

    child = expr.children[0]
    npartitions_in = child.parts.count
    key = get_key_name(expr)
    child_key = get_key_name(child)
    child_dsk = generate_expr_tasks(child, child_ir)

    # Simple all-to-one reduction
    chunk_key = f"chunk-{key}"
    combine_key = f"concat-{key}"
    graph: MutableMapping[tuple[str, int], Any] = {
        (chunk_key, i): (
            _agg_chunk,
            # Fuse with child-expr task
            child_dsk[(child_key, i)],
            expr.request,
            expr.dtype,
        )
        for i in range(npartitions_in)
    }
    graph[(combine_key, 0)] = (_agg_combine, list(graph.keys()))
    graph[(key, 0)] = (
        _agg_finalize,
        (combine_key, 0),
        expr.request,
        expr.dtype,
    )
    return graph
