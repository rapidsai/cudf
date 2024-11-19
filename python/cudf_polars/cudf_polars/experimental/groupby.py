# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel GroupBy Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.dsl.expr import Agg, BinOp, Col, NamedExpr
from cudf_polars.dsl.ir import GroupBy, Select
from cudf_polars.experimental.parallel import (
    PartitionInfo,
    _concat,
    _ir_parts_info,
    _partitionwise_ir_parts_info,
    _partitionwise_ir_tasks,
    generate_ir_tasks,
    get_key_name,
    ir_parts_info,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from polars import GPUEngine

    from cudf_polars.dsl.ir import IR


class GroupByPart(GroupBy):
    """Partitionwise groupby operation."""


class GroupByTree(GroupBy):
    """Groupby tree-reduction operation."""


class GroupByFinalize(Select):
    """Finalize a groupby aggregation."""


_GB_AGG_SUPPORTED = ("sum", "count", "mean")


def lower_groupby_node(ir: GroupBy, rec) -> IR:
    """Rewrite a GroupBy node with proper partitioning."""
    # Lower children first
    children = [rec(child) for child in ir.children]

    # TODO: Skip lowering for single-partition child
    # (Still want to test this case for now)

    # Check that we are groupbing on element-wise
    # keys (is this already guaranteed?)
    for ne in ir.keys:
        if not isinstance(ne.value, Col):
            return ir.reconstruct(children)

    name_map: MutableMapping[str, Any] = {}
    agg_requests_pwise = []
    agg_requests_tree = []
    for ne in ir.agg_requests:
        if not isinstance(ne.value, Agg):
            return ir.reconstruct(children)

        agg = ne.value
        if agg.name not in _GB_AGG_SUPPORTED:
            return ir.reconstruct(children)

        if len(agg.children) > 1:
            return ir.reconstruct(children)

        name = ne.name
        for child in agg.children:
            if not isinstance(child, Col) or child.name != name:
                return ir.reconstruct(children)

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
    return GroupByFinalize(
        schema,
        output_exprs,
        should_broadcast,
        gb_tree,
    )


@_ir_parts_info.register(GroupByPart)
def _(ir: GroupByPart) -> PartitionInfo:
    return _partitionwise_ir_parts_info(ir)


@generate_ir_tasks.register(GroupByPart)
def _(ir: GroupByPart, config: GPUEngine) -> MutableMapping[Any, Any]:
    return _partitionwise_ir_tasks(ir, config)


@_ir_parts_info.register(GroupByTree)
def _(ir: GroupByTree) -> PartitionInfo:
    return PartitionInfo(count=1)


@generate_ir_tasks.register(GroupByTree)
def _(ir: GroupByTree, config: GPUEngine) -> MutableMapping[Any, Any]:
    child = ir.children[0]
    child_count = ir_parts_info(child).count
    child_name = get_key_name(child)
    name = get_key_name(ir)

    # Simple all-to-one reduction.
    # TODO: Use proper tree reduction
    return {
        (name, 0): (
            ir.do_evaluate,
            config,
            *ir._non_child_args,
            (
                _concat,
                [(child_name, i) for i in range(child_count)],
            ),
        )
    }


@_ir_parts_info.register(GroupByFinalize)
def _(ir: GroupByFinalize) -> PartitionInfo:
    return _partitionwise_ir_parts_info(ir)


@generate_ir_tasks.register(GroupByFinalize)
def _(ir: GroupByFinalize, config: GPUEngine) -> MutableMapping[Any, Any]:
    # TODO: Fuse with GroupByTree child task?
    return _partitionwise_ir_tasks(ir, config)
