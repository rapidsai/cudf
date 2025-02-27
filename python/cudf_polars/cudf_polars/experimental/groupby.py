# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel GroupBy Logic."""

from __future__ import annotations

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
        for c, f in ir.config_options.get("executor_options", {})
        .get("cardinality_factor", {})
        .items()
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

    name_map: MutableMapping[str, Any] = {}
    agg_tree: Cast | Agg | None = None
    agg_requests_pwise = []  # Partition-wise requests
    agg_requests_tree = []  # Tree-node requests
    unary_ops: dict[str, dict[str, Any]] = {}

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
        elif isinstance(agg, (Agg, UnaryFunction)):
            if (
                isinstance(agg, UnaryFunction)
                and agg.is_pointwise
                and isinstance(agg.children[0], Agg)
            ):
                # TODO: Handle sequential unary ops
                unary_ops[name] = {"name": agg.name, "options": agg.options}
                agg = agg.children[0]

            if agg.name not in _GB_AGG_SUPPORTED:
                raise NotImplementedError(
                    "GroupBy does not support multiple partitions "
                    f"for this expression:\n{agg}"
                )

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
            # Unsupported expression
            raise NotImplementedError(
                "GroupBy does not support multiple partitions "
                f"for this expression:\n{agg}"
            )  # pragma: no cover

    # Partition-wise groupby operation
    gb_pwise = GroupBy(
        ir.schema,
        ir.keys,
        agg_requests_pwise,
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
        shuffle_options: dict[str, Any] = {}
        gb_inter = Shuffle(
            ir.schema,
            ir.keys,
            shuffle_options,
            gb_pwise,
        )
        partition_info[gb_inter] = PartitionInfo(count=post_aggregation_count)

    # Tree reduction if post_aggregation_count==1
    # (Otherwise, this is another partition-wise op)
    gb_reduce = GroupBy(
        ir.schema,
        ir.keys,
        agg_requests_tree,
        ir.maintain_order,
        ir.options,
        ir.config_options,
        gb_inter,
    )
    partition_info[gb_reduce] = PartitionInfo(count=post_aggregation_count)

    schema = ir.schema
    output_exprs = []
    col_expr: Col | BinOp | UnaryFunction
    for name, dtype in schema.items():
        agg_mapping = name_map.get(name, None)
        if agg_mapping is None:
            col_expr = Col(dtype, name)
        elif "mean" in agg_mapping:
            mean_cols = agg_mapping["mean"]
            col_expr = BinOp(
                dtype,
                plc.binaryop.BinaryOperator.DIV,
                Col(dtype, mean_cols["sum"]),
                Col(dtype, mean_cols["count"]),
            )
        if name in unary_ops:
            col_expr = UnaryFunction(
                dtype,
                unary_ops[name]["name"],
                unary_ops[name]["options"],
                col_expr,
            )
        output_exprs.append(NamedExpr(name, col_expr))
    should_broadcast: bool = False
    new_node = Select(
        schema,
        output_exprs,
        should_broadcast,
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
    n_ary = ir.config_options.get("executor_options", {}).get("groupby_n_ary", 32)
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
