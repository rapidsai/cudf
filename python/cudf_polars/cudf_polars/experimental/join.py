# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Join Logic."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

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


def _split_partition(df: DataFrame, on: list[str], count: int) -> dict[int, DataFrame]:
    on_ind = [df.column_names.index(col) for col in on]
    table, indices = plc.partitioning.hash_partition(df.table, on_ind, count)
    indices += [df.num_rows]
    return {
        i: DataFrame.from_table(
            plc.copying.slice(table, indices[i : i + 2])[0],
            df.column_names,
        )
        for i in range(count)
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
        other_on = [v.name for v in ir.right_on]
    else:
        bcast_name = get_key_name(right)
        bcast_size = partition_info[right].count
        other = get_key_name(left)
        other_on = [v.name for v in ir.left_on]

    graph: MutableMapping[Any, Any] = {}

    # Special handling until RearrangeByColumn is implemented.
    if bcast_size > 1:
        shuffle_name = "shuffle-" + bcast_name
        graph[(shuffle_name, 0)] = (
            _concat,
            [(bcast_name, i) for i in range(bcast_size)],
        )
        bcast_size = 1
        bcast_name = shuffle_name

    out_name = get_key_name(ir)
    out_size = partition_info[ir].count
    split_name = "split-" + out_name
    inter_name = "inter-" + out_name

    for part_out in range(out_size):
        if how != "inner":
            graph[(split_name, part_out)] = (
                _split_partition,
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
