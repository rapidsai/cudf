# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Join Logic."""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import Join
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from collections.abc import MutableMapping

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
    left = Shuffle(
        left.schema,
        ir.left_on,
        shuffle_options,
        left,
    )
    partition_info[left] = PartitionInfo(count=output_count)
    right = Shuffle(
        right.schema,
        ir.right_on,
        shuffle_options,
        right,
    )
    partition_info[right] = PartitionInfo(count=output_count)
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

    output_count = max(partition_info[c].count for c in children)
    if output_count == 1:
        new_node = ir.reconstruct(children)
        partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info
    elif ir.options[0] == "cross":
        raise NotImplementedError(
            "cross join not support for multiple partitions."
        )  # pragma: no cover

    # Hash join
    return _make_hash_join(
        ir,
        output_count,
        partition_info,
        *children,
    )
