# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Partitioned LogicalPlan nodes."""

from __future__ import annotations

import warnings
from functools import partial, singledispatch
from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.dsl.expr import Agg, BinOp, Col, NamedExpr
from cudf_polars.dsl.ir import Scan, Select, broadcast
from cudf_polars.dsl.nodebase import PartitionInfo
from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping, Sequence

    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.nodebase import Node


def get_key_name(node: Node | NamedExpr) -> str:
    """Generate the key name for a Node."""
    if isinstance(node, NamedExpr):
        return f"named-{get_key_name(node.value)}"
    return f"{type(node).__name__.lower()}-{hash(node)}"


@singledispatch
def ir_parts_info(ir: IR) -> PartitionInfo:
    """Return the partitioning info for an IR node."""
    count = 1
    if ir.children:
        count = max(child.parts.count for child in ir.children)
    if count > 1:
        warnings.warn(
            f"Multi-partition support is not implemented for {type(ir)}. "
            f"Partitions will be concatenated. Expect poor performance.",
            stacklevel=2,
        )
    return PartitionInfo(count=1)


@singledispatch
def expr_parts_info(expr: Expr | NamedExpr, child_ir: IR) -> PartitionInfo:
    """
    Return the partitioning info for an Expr.

    Since the partitioning of a leaf Expr depends on
    the child IR node, a `child_ir` positional argument
    is also required.
    """
    if isinstance(expr, NamedExpr):
        return expr_parts_info(expr.value, child_ir)
    if expr.children:
        count = max([expr_parts_info(child, child_ir).count for child in expr.children])
    else:
        count = child_ir.parts.count
    if count > 1:
        warnings.warn(
            f"Multi-partition support is not implemented for {type(expr)}. "
            f"Partitions will be concatenated. Expect poor performance.",
            stacklevel=2,
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
## Node-specific Dispatch Logic
##


##
## NamedExpr
##


def _rename_column(column: Column, name: str):
    return column.rename(name)


@generate_expr_tasks.register(NamedExpr)
def _(expr: NamedExpr, child_ir: IR) -> MutableMapping[Any, Any]:
    # Special case: The graph of a NamedExpr
    # will simply rename the column returned
    # by its value attribute.
    graph = generate_expr_tasks(expr.value, child_ir)
    named_expr_key_name = get_key_name(expr)
    expr_key_name = get_key_name(expr.value)
    for i in range(expr_parts_info(expr.value, child_ir).count):
        graph[(named_expr_key_name, i)] = (
            _rename_column,
            graph.pop((expr_key_name, i)),
            expr.name,
        )
    return graph


##
## Scan
##

_SCAN_SUPPORTED = ("parquet",)


@ir_parts_info.register(Scan)
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


@ir_parts_info.register(Select)
def _(ir: Select) -> PartitionInfo:
    # Partitioning depends on the expression
    df = ir.children[0]
    column_partition_counts = [expr_parts_info(expr, df).count for expr in ir.exprs]
    count = max(column_partition_counts)
    if count > 1:
        warnings.warn(
            f"Multi-partition support is not implemented for {type(ir)}. "
            f"Partitions will be concatenated. Expect poor performance.",
            stacklevel=2,
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
## Col
##


@expr_parts_info.register(Col)
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
## BinOp
##


@expr_parts_info.register(BinOp)
def _(expr: BinOp, child_ir: IR) -> PartitionInfo:
    counts = {expr_parts_info(child, child_ir).count for child in expr.children}
    if len(counts) != 1:
        raise NotImplementedError("Mismatched partition counts.")
    count = counts.pop()
    return PartitionInfo(count=count)


def _binop(left: Column, right: Column, op: Callable, dtype: plc.DataType) -> Column:
    lop = left.obj
    rop = right.obj
    if left.obj.size() != right.obj.size():
        if left.is_scalar:
            lop = left.obj_scalar
        elif right.is_scalar:
            rop = right.obj_scalar
    return Column(
        plc.binaryop.binary_operation(lop, rop, op, dtype),
    )


@generate_expr_tasks.register(BinOp)
def _(expr: BinOp, child_ir: IR) -> MutableMapping[Any, Any]:
    name = get_key_name(expr)
    left = expr.children[0]
    left_name = get_key_name(left)
    left_graph = generate_expr_tasks(left, child_ir)
    right = expr.children[1]
    right_name = get_key_name(right)
    right_graph = generate_expr_tasks(right, child_ir)
    graph = {
        (name, i): (
            _binop,
            left_graph.pop((left_name, i)),
            right_graph.pop((right_name, i)),
            expr.op,
            expr.dtype,
        )
        for i in range(expr_parts_info(left, child_ir).count)
    }
    graph.update(left_graph)
    graph.update(right_graph)
    return graph


##
## Agg
##

_AGG_SUPPORTED = (
    "min",
    "max",
    "first",
    "last",
    "sum",
    "count",
    "mean",
)


@expr_parts_info.register(Agg)
def _(expr: Agg, child_ir: IR) -> PartitionInfo:
    if expr.children:
        count = max([expr_parts_info(child, child_ir).count for child in expr.children])
    else:
        count = child_ir.parts.count
    if count > 1 and expr.name not in _AGG_SUPPORTED:
        assert 0 == 1
        # Only support sum reductions for now.
        warnings.warn(
            f"Multi-partition support is not implemented for {type(expr)}. "
            f"Partitions will be concatenated. Expect poor performance.",
            stacklevel=2,
        )
    return PartitionInfo(count=1)


def _tree_agg(
    op: Callable,
    input: Column | Sequence[Column],
) -> Column:
    if isinstance(input, Column):
        column = input
    elif len(input) == 1:
        column = input[0]
    else:
        column = Column(plc.concatenate.concatenate([col.obj for col in input]))
    return op(column)


def _tree_agg_multi(
    ops: Mapping[str, Callable],
    input: Column | Sequence[DataFrame],
) -> DataFrame:
    if isinstance(input, Column):
        columns = [op(input).rename(name) for name, op in ops.items()]
    else:
        df = DataFrame.concat(input)
        columns = [
            op(df.select_columns({name})[0]).rename(name) for name, op in ops.items()
        ]
    return DataFrame(columns)


def _finalize_mean(df: DataFrame, dtype: plc.DataType) -> Column:
    _sum = df.select_columns({"sum"})[0]
    _count = df.select_columns({"count"})[0]
    return Column(
        plc.binaryop.binary_operation(
            _sum.obj,
            _count.obj,
            plc.binaryop.BinaryOperator.DIV,
            dtype,
        )
    )


@generate_expr_tasks.register(Agg)
def _(expr: Agg, child_ir: IR) -> MutableMapping[Any, Any]:
    if expr.name not in _AGG_SUPPORTED:
        raise NotImplementedError(f"Cannot generate tasks for {expr}.")

    child = expr.children[0]
    npartitions_in = expr_parts_info(child, child_ir).count
    key = get_key_name(expr)
    child_key = get_key_name(child)
    child_dsk = generate_expr_tasks(child, child_ir)

    # Single input-partition shortcut
    if npartitions_in == 1:
        return {
            (key, 0): (
                expr.op,
                child_dsk.pop((child_key, 0)),
            )
        }

    # Check for simple case
    # TODO: Avoid generating entire child_dsk graph?
    if expr.name in ("first", "last"):
        if expr.name == "last":
            index = npartitions_in - 1
        else:
            index = 0
        return {
            (key, 0): (
                _tree_agg,
                expr.op,
                # Fuse with child-expr task
                child_dsk.pop((child_key, index)),
            )
        }

    # Tree func is different for "complex" aggs
    # (Probably a better way to generalize this)
    chunk_func: Callable
    tree_func: Callable
    finalize: Callable | None = None
    chunk_op: Callable | MutableMapping[str, Callable]
    tree_op: Callable | MutableMapping[str, Callable]
    if expr.name == "mean":
        chunk_func = tree_func = _tree_agg_multi
        finalize = _finalize_mean
        chunk_op = {
            "sum": partial(expr._reduce, request=plc.aggregation.sum()),
            "count": expr._count,
        }
        tree_op = {
            "sum": partial(expr._reduce, request=plc.aggregation.sum()),
            "count": partial(expr._reduce, request=plc.aggregation.sum()),
        }
    else:
        chunk_func = tree_func = _tree_agg
        if expr.name == "count":
            # After the initial count operations,
            # we just want to apply a sum aggregation
            chunk_op = expr.op
            tree_op = partial(expr._reduce, request=plc.aggregation.sum())
        else:
            chunk_op = expr.op
            tree_op = expr.op

    # Simple all-to-one reduction
    # TODO: Add proper tree reduction
    tree_key: str = f"tree-{key}"
    combine_key: str = f"combine-{key}"
    graph: MutableMapping[tuple[str, int], Any] = {
        (tree_key, i): (
            chunk_func,
            chunk_op,
            # Fuse with child-expr task
            child_dsk.pop((child_key, i)),
        )
        for i in range(npartitions_in)
    }
    graph[(combine_key, 0)] = (
        tree_func,
        tree_op,
        list(graph.keys()),
    )
    if finalize:
        graph[(key, 0)] = (
            finalize,
            graph.pop((combine_key, 0)),
            expr.dtype,
        )
    else:
        graph[(key, 0)] = graph.pop((combine_key, 0))
    return graph
