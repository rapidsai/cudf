# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Select Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import Select
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


@lower_ir_node.register(Select)
def _(
    ir: Select, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    (child,) = ir.children
    child, partition_info = rec(child)
    new_node = ir.reconstruct([child])

    # Search the expression graph for "complex" operations
    if not all(e.is_pointwise for e in ir.exprs):
        if partition_info[child].count > 1:
            # TODO: Handle non-pointwise expressions.
            raise NotImplementedError(
                f"Selection {ir} does not support multiple partitions."
            )
        else:  # pragma: no cover
            partition_info[new_node] = PartitionInfo(count=1)
            return new_node, partition_info

    # Remaining Select ops are partition-wise
    partition_info[new_node] = PartitionInfo(count=partition_info[child].count)
    return new_node, partition_info
