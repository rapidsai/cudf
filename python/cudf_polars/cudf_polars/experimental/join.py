# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Join Logic."""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import Join
from cudf_polars.experimental.base import PartitionInfo, _concat, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.experimental.shuffle import Shuffle, _partition_dataframe

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


def _maybe_shuffle_frame(
    frame: IR,
    on: tuple[NamedExpr, ...],
    partition_info: MutableMapping[IR, PartitionInfo],
    shuffle_options: dict[str, Any],
    output_count: int,
) -> IR:
    # Shuffle `frame` if it isn't already shuffled.
    if (
        partition_info[frame].partitioned_on == on
        and partition_info[frame].count == output_count
    ):
        # Already shuffled
        return frame
    else:
        # Insert new Shuffle node
        frame = Shuffle(
            frame.schema,
            on,
            shuffle_options,
            frame,
        )
        partition_info[frame] = PartitionInfo(
            count=output_count,
            partitioned_on=on,
        )
        return frame


def _make_hash_join(
    ir: Join,
    output_count: int,
    partition_info: MutableMapping[IR, PartitionInfo],
    left: IR,
    right: IR,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Shuffle left and right dataframes (if necessary)
    shuffle_options: dict[str, Any] = {}  # Unused for now
    new_left = _maybe_shuffle_frame(
        left,
        ir.left_on,
        partition_info,
        shuffle_options,
        output_count,
    )
    new_right = _maybe_shuffle_frame(
        right,
        ir.right_on,
        partition_info,
        shuffle_options,
        output_count,
    )
    if left != new_left or right != new_right:
        ir = ir.reconstruct([new_left, new_right])
    left = new_left
    right = new_right

    # Record new partitioning info
    partitioned_on: tuple[NamedExpr, ...] = ()
    if ir.left_on == ir.right_on or (ir.options[0] in ("Left", "Semi", "Anti")):
        partitioned_on = ir.left_on
    elif ir.options[0] == "Right":
        partitioned_on = ir.right_on
    partition_info[ir] = PartitionInfo(
        count=output_count,
        partitioned_on=partitioned_on,
    )

    return ir, partition_info


def _should_bcast_join(
    ir: Join,
    left: IR,
    right: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    output_count: int,
) -> bool:
    # Decide if a broadcast join is appropriate.
    if partition_info[left].count >= partition_info[right].count:
        small_count = partition_info[right].count
        large = left
        large_on = ir.left_on
    else:
        small_count = partition_info[left].count
        large = right
        large_on = ir.right_on

    # Avoid the broadcast if the "large" table is already shuffled
    large_shuffled = (
        partition_info[large].partitioned_on == large_on
        and partition_info[large].count == output_count
    )

    # Broadcast-Join Criteria:
    # 1. Large dataframe isn't already shuffled
    # 2. Small dataframe has 8 partitions (or fewer).
    #    TODO: Make this value/heuristic configurable).
    #    We may want to account for the number of workers.
    # 3. The "kind" of join is compatible with a broadcast join
    return (
        not large_shuffled
        and small_count <= 8  # TODO: Make this configurable
        and (
            ir.options[0] == "Inner"
            or (ir.options[0] in ("Left", "Semi", "Anti") and large == left)
            or (ir.options[0] == "Right" and large == right)
        )
    )


def _make_bcast_join(
    ir: Join,
    output_count: int,
    partition_info: MutableMapping[IR, PartitionInfo],
    left: IR,
    right: IR,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if ir.options[0] != "Inner":
        shuffle_options: dict[str, Any] = {}
        left_count = partition_info[left].count
        right_count = partition_info[right].count

        # Shuffle the smaller table (if necessary) - Notes:
        # - We need to shuffle the smaller table if
        #   (1) we are not doing an "inner" join,
        #   and (2) the small table contains multiple
        #   partitions.
        # - We cannot simply join a large-table partition
        #   to each small-table partition, and then
        #   concatenate the partial-join results, because
        #   a non-"inner" join does NOT commute with
        #   concatenation.
        # - In some cases, we can perform the partial joins
        #   sequentially. However, we are starting with a
        #   catch-all algorithm that works for all cases.
        if left_count >= right_count:
            right = _maybe_shuffle_frame(
                right,
                ir.right_on,
                partition_info,
                shuffle_options,
                right_count,
            )
        else:
            left = _maybe_shuffle_frame(
                left,
                ir.left_on,
                partition_info,
                shuffle_options,
                left_count,
            )

    new_node = ir.reconstruct([left, right])
    partition_info[new_node] = PartitionInfo(count=output_count)
    return new_node, partition_info


@lower_ir_node.register(Join)
def _(
    ir: Join, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    left, right = children
    output_count = max(partition_info[left].count, partition_info[right].count)
    if output_count == 1:
        new_node = ir.reconstruct(children)
        partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info
    elif ir.options[0] == "Cross":
        raise NotImplementedError(
            "Cross join not support for multiple partitions."
        )  # pragma: no cover

    if _should_bcast_join(ir, left, right, partition_info, output_count):
        # Create a broadcast join
        return _make_bcast_join(
            ir,
            output_count,
            partition_info,
            left,
            right,
        )
    else:
        # Create a hash join
        return _make_hash_join(
            ir,
            output_count,
            partition_info,
            left,
            right,
        )


@generate_ir_tasks.register(Join)
def _(
    ir: Join, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    left, right = ir.children
    output_count = partition_info[ir].count

    left_partitioned = (
        partition_info[left].partitioned_on == ir.left_on
        and partition_info[left].count == output_count
    )
    right_partitioned = (
        partition_info[right].partitioned_on == ir.right_on
        and partition_info[right].count == output_count
    )

    if output_count == 1 or (left_partitioned and right_partitioned):
        # Partition-wise join
        left_name = get_key_name(left)
        right_name = get_key_name(right)
        return {
            key: (
                ir.do_evaluate,
                *ir._non_child_args,
                (left_name, i),
                (right_name, i),
            )
            for i, key in enumerate(partition_info[ir].keys(ir))
        }
    else:
        # Broadcast join
        left_parts = partition_info[left]
        right_parts = partition_info[right]
        if left_parts.count >= right_parts.count:
            small_side = "Right"
            small_name = get_key_name(right)
            small_size = partition_info[right].count
            large_name = get_key_name(left)
            large_on = ir.left_on
        else:
            small_side = "Left"
            small_name = get_key_name(left)
            small_size = partition_info[left].count
            large_name = get_key_name(right)
            large_on = ir.right_on

        graph: MutableMapping[Any, Any] = {}

        out_name = get_key_name(ir)
        out_size = partition_info[ir].count
        split_name = f"split-{out_name}"
        inter_name = f"inter-{out_name}"

        for part_out in range(out_size):
            if ir.options[0] != "Inner":
                graph[(split_name, part_out)] = (
                    _partition_dataframe,
                    (large_name, part_out),
                    large_on,
                    small_size,
                )

            _concat_list = []
            for j in range(small_size):
                join_children = [
                    (
                        (
                            operator.getitem,
                            (split_name, part_out),
                            j,
                        )
                        if ir.options[0] != "Inner"
                        else (large_name, part_out)
                    ),
                    (small_name, j),
                ]
                if small_side == "Left":
                    join_children.reverse()

                inter_key = (inter_name, part_out, j)
                graph[(inter_name, part_out, j)] = (
                    ir.do_evaluate,
                    ir.left_on,
                    ir.right_on,
                    ir.options,
                    *join_children,
                )
                _concat_list.append(inter_key)
            if len(_concat_list) == 1:
                graph[(out_name, part_out)] = graph.pop(_concat_list[0])
            else:
                graph[(out_name, part_out)] = (_concat, _concat_list)

        return graph
