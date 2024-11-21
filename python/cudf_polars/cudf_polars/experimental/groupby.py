# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel GroupBy Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.dsl.expr import Agg, BinOp, Cast, Col, Len, NamedExpr
from cudf_polars.dsl.ir import GroupBy, Select
from cudf_polars.experimental.parallel import (
    PartitionInfo,
    _concat,
    _default_lower_ir_node,
    _lower_children,
    _partitionwise_ir_tasks,
    generate_ir_tasks,
    get_key_name,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


class GroupByPart(GroupBy):
    """Partitionwise groupby operation."""


class GroupByTree(GroupBy):
    """Groupby tree-reduction operation."""


class GroupByFinalize(Select):
    """Finalize a groupby aggregation."""


_GB_AGG_SUPPORTED = ("sum", "count", "mean")


def lower_groupby_node(
    ir: GroupBy, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Rewrite a GroupBy node with proper partitioning."""
    # Lower children first
    children, partition_info = _lower_children(ir, rec)

    if partition_info[children[0]].count == 1:
        # Single partition
        return _default_lower_ir_node(ir, rec)

    # Check that we are groupbing on element-wise
    # keys (is this already guaranteed?)
    for ne in ir.keys:
        if not isinstance(ne.value, Col):
            return _default_lower_ir_node(ir, rec)

    name_map: MutableMapping[str, Any] = {}
    agg_tree: Cast | Agg | None = None
    agg_requests_pwise = []
    agg_requests_tree = []
    for ne in ir.agg_requests:
        name = ne.name
        agg = ne.value
        if isinstance(agg, Cast) and isinstance(agg.children[0], Len):
            # Len
            agg_requests_pwise.append(ne)
            agg_tree = Cast(
                agg.dtype, Agg(agg.dtype, "sum", None, Col(agg.dtype, name))
            )
            agg_requests_tree.append(NamedExpr(name, agg_tree))
        elif isinstance(agg, Agg):
            # Agg
            if agg.name not in _GB_AGG_SUPPORTED:
                return _default_lower_ir_node(ir, rec)

            if len(agg.children) > 1:
                return _default_lower_ir_node(ir, rec)

            if agg.name == "sum":
                # Partwise
                agg_pwise = Agg(agg.dtype, "sum", agg.options, *agg.children)
                agg_requests_pwise.append(NamedExpr(name, agg_pwise))
                # Tree
                agg_tree = Agg(agg.dtype, "sum", agg.options, Col(agg.dtype, name))
                agg_requests_tree.append(NamedExpr(name, agg_tree))
            elif agg.name == "count":
                # Partwise
                agg_pwise = Agg(agg.dtype, "count", agg.options, *agg.children)
                agg_requests_pwise.append(NamedExpr(name, agg_pwise))
                # Tree
                agg_tree = Agg(agg.dtype, "sum", agg.options, Col(agg.dtype, name))
                agg_requests_tree.append(NamedExpr(name, agg_tree))
            elif agg.name == "mean":
                name_map[name] = {agg.name: {}}
                for sub in ["sum", "count"]:
                    # Partwise
                    tmp_name = f"{name}__{sub}"
                    name_map[name][agg.name][sub] = tmp_name
                    agg_pwise = Agg(agg.dtype, sub, agg.options, *agg.children)
                    agg_requests_pwise.append(NamedExpr(tmp_name, agg_pwise))
                    # Tree
                    child = Col(agg.dtype, tmp_name)
                    agg_tree = Agg(agg.dtype, "sum", agg.options, child)
                    agg_requests_tree.append(NamedExpr(tmp_name, agg_tree))
        else:
            # Unsupported
            return _default_lower_ir_node(ir, rec)

    gb_pwise = GroupByPart(
        ir.schema,
        ir.keys,
        agg_requests_pwise,
        ir.maintain_order,
        ir.options,
        *children,
    )

    gb_tree = GroupByTree(
        ir.schema,
        ir.keys,
        agg_requests_tree,
        ir.maintain_order,
        ir.options,
        gb_pwise,
    )

    schema = ir.schema
    output_exprs = []
    for name, dtype in schema.items():
        agg_mapping = name_map.get(name, None)
        if agg_mapping is None:
            output_exprs.append(NamedExpr(name, Col(dtype, name)))
        elif "mean" in agg_mapping:
            mean_cols = agg_mapping["mean"]
            output_exprs.append(
                NamedExpr(
                    name,
                    BinOp(
                        dtype,
                        plc.binaryop.BinaryOperator.DIV,
                        Col(dtype, mean_cols["sum"]),
                        Col(dtype, mean_cols["count"]),
                    ),
                )
            )
    should_broadcast: bool = False
    new_node = GroupByFinalize(
        schema,
        output_exprs,
        should_broadcast,
        gb_tree,
    )
    partition_info[new_node] = PartitionInfo(count=1)
    return new_node, partition_info


@generate_ir_tasks.register(GroupByPart)
def _(
    ir: GroupByPart, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    return _partitionwise_ir_tasks(ir, partition_info)


def _tree_node(do_evaluate, batch, *args):
    return do_evaluate(*args, _concat(batch))


@generate_ir_tasks.register(GroupByTree)
def _(
    ir: GroupByTree, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    child = ir.children[0]
    child_count = partition_info[child].count
    child_name = get_key_name(child)
    name = get_key_name(ir)

    # Simple tree reduction.
    j = 0
    graph: MutableMapping[Any, Any] = {}
    split_every = 32
    keys: list[Any] = [(child_name, i) for i in range(child_count)]
    while len(keys) > split_every:
        new_keys: list[Any] = []
        for i, k in enumerate(range(0, len(keys), split_every)):
            batch = keys[k : k + split_every]
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


@generate_ir_tasks.register(GroupByFinalize)
def _(
    ir: GroupByFinalize, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    # TODO: Fuse with GroupByTree child task?
    return _partitionwise_ir_tasks(ir, partition_info)
