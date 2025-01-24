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


def _make_hash_join(
    ir: Join,
    output_count: int,
    partition_info: MutableMapping[IR, PartitionInfo],
    left: IR,
    right: IR,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    shuffle_options: dict[str, Any] = {}

    if not (
        partition_info[left].partitioned_on == ir.left_on
        and partition_info[left].count == output_count
    ):
        left = Shuffle(
            left.schema,
            ir.left_on,
            shuffle_options,
            left,
        )
        partition_info[left] = PartitionInfo(
            count=output_count,
            partitioned_on=ir.left_on,
        )

    if not (
        partition_info[right].partitioned_on == ir.right_on
        and partition_info[right].count == output_count
    ):
        right = Shuffle(
            right.schema,
            ir.right_on,
            shuffle_options,
            right,
        )
        partition_info[right] = PartitionInfo(
            count=output_count,
            partitioned_on=ir.right_on,
        )

    new_node = ir.reconstruct([left, right])

    partitioned_on: tuple[NamedExpr, ...] = ()
    if ir.left_on == ir.right_on or (ir.options[0] in ("left", "semi", "anti")):
        partitioned_on = ir.left_on
    elif ir.options[0] == "right":  # pragma: no cover
        partitioned_on = ir.right_on
    partition_info[new_node] = PartitionInfo(
        count=output_count,
        partitioned_on=partitioned_on,
    )

    return new_node, partition_info


def _make_bcast_join(
    ir: Join,
    output_count: int,
    partition_info: MutableMapping[IR, PartitionInfo],
    left: IR,
    right: IR,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    how = ir.options[0]
    if how != "inner":
        shuffle_options: dict[str, Any] = {}
        left_count = partition_info[left].count
        right_count = partition_info[right].count
        if left_count >= right_count:
            right = Shuffle(
                right.schema,
                ir.right_on,
                shuffle_options,
                right,
            )
            partition_info[right] = PartitionInfo(
                count=right_count,
                partitioned_on=ir.right_on,
            )
        else:
            left = Shuffle(
                left.schema,
                ir.left_on,
                shuffle_options,
                left,
            )
            partition_info[left] = PartitionInfo(
                count=left_count,
                partitioned_on=ir.left_on,
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

    how = ir.options[0]
    left, right = children
    output_count = max(partition_info[left].count, partition_info[right].count)
    if output_count == 1:
        new_node = ir.reconstruct(children)
        partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info
    elif how == "cross":
        raise NotImplementedError(
            "cross join not support for multiple partitions."
        )  # pragma: no cover

    # Use a broadcast join if appropriate
    if partition_info[left].count >= partition_info[right].count:
        bcast_count = partition_info[right].count
        other = left
        other_on = ir.left_on
    else:
        bcast_count = partition_info[left].count
        other = right
        other_on = ir.right_on

    # Avoid the broadcast if the "large" table is already shuffled
    other_shuffled = (
        partition_info[other].partitioned_on == other_on
        and partition_info[other].count == output_count
    )

    if (
        not other_shuffled
        and bcast_count <= 8  # TODO: Make this configurable
        and (
            how == "inner"
            or (how in ("left", "semi", "anti") and other == left)
            or (how == "right" and other == right)
        )
    ):
        # Create a broadcast join
        return _make_bcast_join(
            ir,
            output_count,
            partition_info,
            *children,
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
        how = ir.options[0]
        left_parts = partition_info[left]
        right_parts = partition_info[right]
        if left_parts.count >= right_parts.count:
            bcast_side = "right"
            bcast_name = get_key_name(right)
            bcast_size = partition_info[right].count
            other_name = get_key_name(left)
            other_on = ir.left_on
        else:
            bcast_side = "left"
            bcast_name = get_key_name(left)
            bcast_size = partition_info[left].count
            other_name = get_key_name(right)
            other_on = ir.right_on

        graph: MutableMapping[Any, Any] = {}

        out_name = get_key_name(ir)
        out_size = partition_info[ir].count
        split_name = "split-" + out_name
        inter_name = "inter-" + out_name

        for part_out in range(out_size):
            if how != "inner":
                graph[(split_name, part_out)] = (
                    _partition_dataframe,
                    (other_name, part_out),
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
                        else (other_name, part_out)
                    ),
                    (bcast_name, j),
                ]
                if bcast_side == "left":
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
