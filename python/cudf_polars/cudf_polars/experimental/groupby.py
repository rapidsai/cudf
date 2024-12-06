# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel GroupBy Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.dsl.expr import Agg, BinOp, Cast, Col, Len, NamedExpr
from cudf_polars.dsl.ir import GroupBy, Select
from cudf_polars.experimental.base import PartitionInfo, _concat, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


_GB_AGG_SUPPORTED = ("sum", "count", "mean")


def _single_fallback(
    ir: IR,
    child: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    unsupported_agg: Expr | None = None,
):
    if partition_info[child].count > 1:
        msg = f"Class {type(ir)} does not support multiple partitions."
        if unsupported_agg:
            msg = msg[:-1] + f" with {unsupported_agg} expression."
        raise NotImplementedError(msg)
    else:  # pragma: no cover
        new_node = ir.reconstruct([child])
        partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info


@lower_ir_node.register(GroupBy)
def _(
    ir: GroupBy, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    (child,) = ir.children
    child, partition_info = rec(child)

    if partition_info[child].count == 1:
        # Single partition
        return _single_fallback(ir, child, partition_info)

    # Check that we are grouping on element-wise
    # keys (is this already guaranteed?)
    for ne in ir.keys:
        if not isinstance(ne.value, Col):  # pragma: no cover
            return _single_fallback(ir, child, partition_info)

    name_map: MutableMapping[str, Any] = {}
    agg_tree: Cast | Agg | None = None
    agg_requests_pwise = []  # Partition-wise requests
    agg_requests_tree = []  # Tree-node requests

    for ne in ir.agg_requests:
        name = ne.name
        agg: Expr = ne.value
        dtype = agg.dtype
        agg = agg.children[0] if isinstance(agg, Cast) else agg
        if isinstance(agg, Len):
            agg_requests_pwise.append(ne)
            agg_requests_tree.append(
                NamedExpr(
                    name,
                    Cast(
                        dtype,
                        Agg(dtype, "sum", None, Col(dtype, name)),
                    ),
                )
            )
        elif isinstance(agg, Agg):
            if agg.name not in _GB_AGG_SUPPORTED:
                return _single_fallback(ir, child, partition_info, agg)

            if agg.name in ("sum", "count"):
                agg_requests_pwise.append(ne)
                agg_requests_tree.append(
                    NamedExpr(
                        name,
                        Cast(
                            dtype,
                            Agg(dtype, "sum", agg.options, Col(dtype, name)),
                        ),
                    )
                )
            elif agg.name == "mean":
                name_map[name] = {agg.name: {}}
                for sub in ["sum", "count"]:
                    # Partwise
                    tmp_name = f"{name}__{sub}"
                    name_map[name][agg.name][sub] = tmp_name
                    agg_pwise = Agg(dtype, sub, agg.options, *agg.children)
                    agg_requests_pwise.append(NamedExpr(tmp_name, agg_pwise))
                    # Tree
                    agg_tree = Agg(dtype, "sum", agg.options, Col(dtype, tmp_name))
                    agg_requests_tree.append(NamedExpr(tmp_name, agg_tree))
        else:
            # Unsupported
            return _single_fallback(ir, child, partition_info, agg)  # pragma: no cover

    gb_pwise = GroupBy(
        ir.schema,
        ir.keys,
        agg_requests_pwise,
        ir.maintain_order,
        ir.options,
        child,
    )
    child_count = partition_info[child].count
    partition_info[gb_pwise] = PartitionInfo(count=child_count)

    gb_tree = GroupBy(
        ir.schema,
        ir.keys,
        agg_requests_tree,
        ir.maintain_order,
        ir.options,
        gb_pwise,
    )
    partition_info[gb_tree] = PartitionInfo(count=1)

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
    new_node = Select(
        schema,
        output_exprs,
        should_broadcast,
        gb_tree,
    )
    partition_info[new_node] = PartitionInfo(count=1)
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

    # Simple tree reduction.
    j = 0
    graph: MutableMapping[Any, Any] = {}
    split_every = 32
    name = get_key_name(ir)
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
