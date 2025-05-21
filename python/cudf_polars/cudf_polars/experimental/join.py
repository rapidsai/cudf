# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Join Logic."""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import ConditionalJoin, Join
from cudf_polars.experimental.base import PartitionInfo, TableStats, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.shuffle import Shuffle, _partition_dataframe
from cudf_polars.experimental.utils import _concat, _fallback_inform, _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions


def _maybe_shuffle_frame(
    frame: IR,
    on: tuple[NamedExpr, ...],
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
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
            config_options,
            frame,
        )
        partition_info[frame] = PartitionInfo.new(
            frame,
            partition_info,
            count=output_count,
            partitioned_on=on,
        )
        return frame


def _make_hash_join(
    ir: Join,
    output_count: int,
    partition_info: MutableMapping[IR, PartitionInfo],
    estimated_output_rows: int | None,
    left: IR,
    right: IR,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Shuffle left and right dataframes (if necessary)
    new_left = _maybe_shuffle_frame(
        left,
        ir.left_on,
        partition_info,
        ir.config_options,
        output_count,
    )
    new_right = _maybe_shuffle_frame(
        right,
        ir.right_on,
        partition_info,
        ir.config_options,
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
    pi = PartitionInfo.new(
        ir,
        partition_info,
        count=output_count,
        partitioned_on=partitioned_on,
    )
    if pi.table_stats is not None and estimated_output_rows is not None:
        pi.table_stats = TableStats(pi.table_stats.column_stats, estimated_output_rows)
    partition_info[ir] = pi
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
    # 2. Small dataframe meets broadcast_join_limit.
    # 3. The "kind" of join is compatible with a broadcast join
    assert ir.config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_tasks'"
    )

    return (
        not large_shuffled
        and small_count <= ir.config_options.executor.broadcast_join_limit
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
    estimated_output_rows: int | None,
    left: IR,
    right: IR,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if ir.options[0] != "Inner":
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
                ir.config_options,
                right_count,
            )
        else:
            left = _maybe_shuffle_frame(
                left,
                ir.left_on,
                partition_info,
                ir.config_options,
                left_count,
            )

    new_node = ir.reconstruct([left, right])
    pi = PartitionInfo.new(new_node, partition_info, count=output_count)
    if pi.table_stats is not None and estimated_output_rows is not None:
        pi.table_stats = TableStats(pi.table_stats.column_stats, estimated_output_rows)
    partition_info[new_node] = pi
    return new_node, partition_info


@lower_ir_node.register(ConditionalJoin)
def _(
    ir: ConditionalJoin, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if ir.options[2]:  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg="Slice not supported in ConditionalJoin for multiple partitions.",
        )

    # Lower children
    left, right = ir.children
    left, pi_left = rec(left)
    right, pi_right = rec(right)

    # Fallback to single partition on the smaller table
    left_count = pi_left[left].count
    right_count = pi_right[right].count
    output_count = max(left_count, right_count)
    fallback_msg = "ConditionalJoin not supported for multiple partitions."
    if left_count < right_count:
        if left_count > 1:
            left = Repartition(left.schema, left)
            pi_left[left] = PartitionInfo.new(left, pi_left, count=1)
            _fallback_inform(fallback_msg, rec.state["config_options"])
    elif right_count > 1:
        right = Repartition(left.schema, right)
        pi_right[right] = PartitionInfo.new(right, pi_right, count=1)
        _fallback_inform(fallback_msg, rec.state["config_options"])

    # Reconstruct and return
    new_node = ir.reconstruct([left, right])
    partition_info = reduce(operator.or_, (pi_left, pi_right))
    partition_info[new_node] = PartitionInfo.new(
        new_node, partition_info, count=output_count
    )
    return new_node, partition_info


@lower_ir_node.register(Join)
def _(
    ir: Join, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    left, right = children
    left_stats = partition_info[left].table_stats
    right_stats = partition_info[right].table_stats
    if left_stats is not None and right_stats is not None:
        left_rows = left_stats.num_rows
        right_rows = right_stats.num_rows
        left_keys = [e.name for e in ir.left_on]
        right_keys = [e.name for e in ir.right_on]
        unique_counts = -1
        for lname, rname in zip(left_keys, right_keys, strict=True):
            uniques_left, uniques_right = None, None
            if lname in left_stats.column_stats:
                uniques_left = left_stats.column_stats[lname].unique_count
            if rname in right_stats.column_stats:
                uniques_right = right_stats.column_stats[rname].unique_count
            uniques = (uniques_left, uniques_right)
            unique_counts = max(
                unique_counts,
                max(
                    (u for u in uniques if u is not None),
                    default=min(left_rows, right_rows),
                ),
            )
        estimated_output_rows = max(1, left_rows * right_rows // unique_counts)
        max_input_rows = max(left_rows, right_rows)
        join_stats = TableStats(
            TableStats.merge_column_stats(
                left_stats.column_stats, right_stats.column_stats
            ),
            estimated_output_rows,
        )
    else:  # pragma: no cover; We usually have basic table stats (num_rows)
        max_input_rows = None
        estimated_output_rows = None
        join_stats = None

    output_count = max(partition_info[left].count, partition_info[right].count)
    if output_count == 1:
        new_node = ir.reconstruct(children)
        partition_info[new_node] = PartitionInfo.new(
            new_node, partition_info, count=1, table_stats=join_stats
        )
        return new_node, partition_info
    elif ir.options[0] == "Cross":  # pragma: no cover
        return _lower_ir_fallback(
            ir, rec, msg="Cross join not support for multiple partitions."
        )

    if _should_bcast_join(ir, left, right, partition_info, output_count):
        # Create a broadcast join
        joined, partition_info = _make_bcast_join(
            ir,
            output_count,
            partition_info,
            estimated_output_rows,
            left,
            right,
        )
    else:
        # Create a hash join
        joined, partition_info = _make_hash_join(
            ir,
            output_count,
            partition_info,
            estimated_output_rows,
            left,
            right,
        )

    if (
        join_stats is not None
        and estimated_output_rows is not None
        and max_input_rows is not None
        and output_count
        > (
            new_count := max(
                1, int(1 * estimated_output_rows * output_count // max_input_rows)
            )
        )
    ):
        assert ir.config_options.executor.name == "streaming", (
            "'in-memory' executor not supported in 'lower_ir_node'"
        )

        # Only repartition if estimated row count suggests we should
        # go to a smaller number of partitions and the estimated
        # output size also favors a smaller partition count.
        estimated_output_size = (
            sum(
                join_stats.column_stats[col].element_size
                if col in join_stats.column_stats
                else 16
                for col in joined.schema
            )
            * join_stats.num_rows
        )
        target_size = rec.state["config_options"].executor.target_partition_size // 4
        if (ideal_count := estimated_output_size // target_size) > new_count:
            # Throttle repartitioning with a conservative
            # partition-size target (1/4).
            new_count = min(ideal_count, output_count)

        if new_count < output_count:
            result = Repartition(joined.schema, joined)
            partition_info[result] = PartitionInfo.new(
                result, partition_info, count=new_count, table_stats=join_stats
            )
            return result, partition_info

    return joined, partition_info


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
        getit_name = f"getit-{out_name}"
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
                left_key: tuple[str, int] | tuple[str, int, int]
                if ir.options[0] != "Inner":
                    left_key = (getit_name, part_out, j)
                    graph[left_key] = (operator.getitem, (split_name, part_out), j)
                else:
                    left_key = (large_name, part_out)
                join_children = [left_key, (small_name, j)]
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
                graph[(out_name, part_out)] = (_concat, *_concat_list)

        return graph
