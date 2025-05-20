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
    partition_info[ir] = PartitionInfo.new(
        ir,
        partition_info,
        count=output_count,
        partitioned_on=partitioned_on,
        table_stats=_join_table_stats(ir, left, right, partition_info),
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
    partition_info[new_node] = PartitionInfo.new(
        new_node,
        partition_info,
        count=output_count,
        table_stats=_join_table_stats(ir, left, right, partition_info),
    )
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


def _join_table_stats(
    ir: Join,
    left: IR,
    right: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
) -> TableStats | None:
    """Return TableStats for a join operation."""
    if ir.options[0] != "Inner":
        # TODO: Handle other join types
        return None

    # Left
    left_table_stats = partition_info[left].table_stats
    if left_table_stats is None:
        return None  # pragma: no cover
    left_card = left_table_stats.num_rows
    left_on = [ne.name for ne in ir.left_on]
    left_on_unique_counts = [
        stats.unique_count
        for name, stats in left_table_stats.column_stats.items()
        if name in left_on
    ]
    if not left_on_unique_counts:
        return None
    left_tdom = max(min(reduce(operator.mul, left_on_unique_counts), left_card), 1)

    # Right
    right_table_stats = partition_info[right].table_stats
    if right_table_stats is None:
        return None  # pragma: no cover
    right_card = right_table_stats.num_rows
    right_on = [ne.name for ne in ir.right_on]
    right_on_unique_counts = [
        stats.unique_count
        for name, stats in right_table_stats.column_stats.items()
        if name in right_on
    ]
    if not right_on_unique_counts:
        return None  # pragma: no cover
    right_tdom = max(min(reduce(operator.mul, right_on_unique_counts), right_card), 1)

    return TableStats.merge(
        [left_table_stats, right_table_stats],
        num_rows=max(
            int(
                # Ref. S. Ebergen Thesis (2022)
                (left_card * right_card) / min(left_tdom, right_tdom)
            ),
            1,
        ),
    )


def _maybe_repartition_child(
    ir: Join,
    child: IR,
    side: str,
    output_count: int,
    partition_info: MutableMapping[IR, PartitionInfo],
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Repartition a Join child if allowed.

    Parameters
    ----------
    ir
        The Join parent of ``child``.
    child
        The IR node being joined.
    side
        Which side this child corresponds to.
    output_count
        Current output partition count of ``ir``.
    partition_info
        A mapping from unique IR nodes to the associated
        PartitionInfo object.

    Returns
    -------
    child, partition_info
        The new child, and an updated mapping
        from unique IR nodes to associated PartitionInfo objects.

    Notes
    -----
    This function will use TableStats information to estimate
    the size of ``child``. A Repartition node will be added
    to the IR graph if this estimates suggest a smaller
    partition count is appropriate.
    """
    assert ir.config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node'"
    )
    blocksize = ir.config_options.executor.target_partition_size
    broadcast_join_limit = ir.config_options.executor.broadcast_join_limit
    join_on = ir.left_on if side == "left" else ir.right_on
    assert side in ("left", "right"), "Unexpected `side` argument."

    child_count = partition_info[child].count
    shuffled = (
        partition_info[child].partitioned_on == join_on and child_count == output_count
    )
    table_stats = partition_info[child].table_stats
    if not shuffled and child_count < output_count and table_stats is not None:
        row_size = 0
        for name in child.schema:
            if name in table_stats.column_stats:
                row_size += table_stats.column_stats[name].element_size
            else:
                row_size = 0
                break
        if row_size:
            ideal_count = max(1, int(row_size * table_stats.num_rows / blocksize))
            if ideal_count <= broadcast_join_limit and ideal_count < child_count:
                child = Repartition(child.schema, child)
                partition_info[child] = PartitionInfo.new(
                    child,
                    partition_info,
                    count=ideal_count,
                )

    return child, partition_info


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
        partition_info[new_node] = PartitionInfo.new(
            new_node,
            partition_info,
            count=1,
            table_stats=_join_table_stats(ir, left, right, partition_info),
        )
        return new_node, partition_info
    elif ir.options[0] == "Cross":  # pragma: no cover
        return _lower_ir_fallback(
            ir, rec, msg="Cross join not support for multiple partitions."
        )

    # Repartition children if we can
    left, partition_info = _maybe_repartition_child(
        ir, left, "left", output_count, partition_info
    )
    right, partition_info = _maybe_repartition_child(
        ir, right, "right", output_count, partition_info
    )

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
