# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Join Logic."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import pyarrow as pa

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import Join
from cudf_polars.experimental.parallel import (
    _concat,
    _default_lower_ir_node,
    _lower_children,
    generate_ir_tasks,
    get_key_name,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer, PartitionInfo


class BroadcastJoin(Join):
    """Broadcast Join operation."""


class LeftBroadcastJoin(BroadcastJoin):
    """Left Broadcast Join operation."""


class RightBroadcastJoin(BroadcastJoin):
    """Right Broadcast Join operation."""


def lower_join_node(
    ir: Join, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Rewrite a Join node with proper partitioning."""
    # TODO: Add shuffle-based join.
    # (Currently using broadcast join in all cases)

    # Lower children first
    children, partition_info = _lower_children(ir, rec)

    how = ir.options[0]
    if how not in ("inner", "left", "right"):
        # Not supported (yet)
        return _default_lower_ir_node(ir, rec)

    assert len(children) == 2
    left, right = children
    left_parts = partition_info[left]
    right_parts = partition_info[right]
    if left_parts.count == right_parts.count == 1:
        # Single-partition case
        return _default_lower_ir_node(ir, rec)
    elif left_parts.count >= right_parts.count and how in ("inner", "left"):
        # Broadcast right to every partition of left
        new_node = RightBroadcastJoin(
            ir.schema,
            ir.left_on,
            ir.right_on,
            ir.options,
            *children,
        )
        partition_info[new_node] = partition_info[left]
    else:
        # Broadcast left to every partition of right
        new_node = LeftBroadcastJoin(
            ir.schema,
            ir.left_on,
            ir.right_on,
            ir.options,
            *children,
        )
        partition_info[new_node] = partition_info[right]
    return new_node, partition_info


def rearrange_by_column(
    name_out: str,
    name_in: str,
    on: tuple[NamedExpr, ...],
    count_in: int,
    count_out: int,
) -> MutableMapping[Any, Any]:
    """Shuffle on a list of columns."""
    # Simple all-to-all shuffle (for now)
    split_name = f"split-{name_out}"
    inter_name = f"inter-{name_out}"

    graph: MutableMapping[Any, Any] = {}
    for part_out in range(count_out):
        _concat_list = []
        for part_in in range(count_in):
            graph[(split_name, part_in)] = (
                _split_by_column,
                (name_in, part_in),
                on,
                count_out,
            )
            _concat_list.append((inter_name, part_out, part_in))
            graph[_concat_list[-1]] = (
                operator.getitem,
                (split_name, part_in),
                part_out,
            )
        graph[(name_out, part_out)] = (_concat, _concat_list)
    return graph


def _split_by_column(
    df: DataFrame,
    on: tuple[NamedExpr, ...],
    count: int,
) -> dict[int, DataFrame]:
    # Extract the partition-map column
    if len(on) == 1 and on[0].name == "_partitions":
        # The `on` argument already contains the
        # destination partition id for each row.
        partition_map = on[0].evaluate(df).obj
    else:
        # Use murmurhash % count to choose the
        # destination partition id for each row.
        partition_map = plc.binaryop.binary_operation(
            plc.hashing.murmurhash3_x86_32(
                DataFrame([expr.evaluate(df) for expr in on]).table
            ),
            plc.interop.from_arrow(pa.scalar(count, type="uint32")),
            plc.binaryop.BinaryOperator.PYMOD,
            plc.types.DataType(plc.types.TypeId.UINT32),
        )

    # Split and return the partitioned result
    return {
        i: DataFrame.from_table(
            split,
            df.column_names,
        )
        for i, split in enumerate(
            plc.copying.split(
                *plc.partitioning.partition(
                    df.table,
                    partition_map,
                    count,
                )
            )
        )
    }


@generate_ir_tasks.register(BroadcastJoin)
def _(
    ir: BroadcastJoin, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    how = ir.options[0]
    left, right = ir.children
    broadcast_side = "right" if isinstance(ir, RightBroadcastJoin) else "left"
    if broadcast_side == "left":
        bcast_name = get_key_name(left)
        bcast_size = partition_info[left].count
        other = get_key_name(right)
        other_on = ir.right_on
        bcast_on = ir.left_on
    else:
        bcast_name = get_key_name(right)
        bcast_size = partition_info[right].count
        other = get_key_name(left)
        other_on = ir.left_on
        bcast_on = ir.right_on

    graph: MutableMapping[Any, Any] = {}

    # Shuffle broadcasted side if necessary
    if how != "inner" and bcast_size > 1:
        shuffle_name = "shuffle-" + bcast_name
        graph = rearrange_by_column(
            shuffle_name,
            bcast_name,
            bcast_on,
            bcast_size,
            bcast_size,
        )
        bcast_name = shuffle_name

    out_name = get_key_name(ir)
    out_size = partition_info[ir].count
    split_name = "split-" + out_name
    inter_name = "inter-" + out_name

    for part_out in range(out_size):
        if how != "inner":
            graph[(split_name, part_out)] = (
                _split_by_column,
                (other, part_out),
                other_on,
                bcast_size,
            )

        _concat_list = []
        for j in range(bcast_size):
            _merge_args = [
                (
                    (
                        operator.getitem,
                        (split_name, part_out),
                        j,
                    )
                    if how != "inner"
                    else (other, part_out)
                ),
                (bcast_name, j),
            ]
            if broadcast_side == "left":
                _merge_args.reverse()

            inter_key = (inter_name, part_out, j)
            graph[(inter_name, part_out, j)] = (
                ir.do_evaluate,
                ir.left_on,
                ir.right_on,
                ir.options,
                *_merge_args,
            )
            _concat_list.append(inter_key)
        if len(_concat_list) == 1:
            graph[(out_name, part_out)] = graph.pop(_concat_list[0])
        else:
            graph[(out_name, part_out)] = (_concat, _concat_list)

    return graph
