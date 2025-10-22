# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core lowering logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cudf_polars.experimental.rapidsmpf.io  # noqa: F401
from cudf_polars.dsl.ir import IR, Sort
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.io import StreamingSink
from cudf_polars.experimental.parallel import _lower_ir_pwise
from cudf_polars.experimental.rapidsmpf.dispatch import (
    lower_ir_node,
)
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.sort import ShuffleSorted
from cudf_polars.experimental.utils import _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.experimental.rapidsmpf.dispatch import LowerIRTransformer


@lower_ir_node.register(IR)
def _lower_ir_node_task_engine(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Use task-engine lowering logic
    from cudf_polars.experimental.dispatch import lower_ir_node as base_lower_ir_node

    return base_lower_ir_node(ir, rec)


@lower_ir_node.register(ShuffleSorted)
@lower_ir_node.register(StreamingSink)
def _unsupported(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Unsupported operations - Fall back to a single partition/chunk.
    return _lower_ir_fallback(
        ir, rec, msg=f"Class {type(ir)} does not support multiple partitions."
    )


@lower_ir_node.register(Sort)
def _(
    ir: Sort, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if ir.zlice is not None:
        # Top- or bottom-k support
        has_offset = ir.zlice[0] > 0 or (
            ir.zlice[0] < 0
            and ir.zlice[1] is not None
            and ir.zlice[0] + ir.zlice[1] < 0
        )
        if not has_offset:
            # Sort input partitions
            new_node, partition_info = _lower_ir_pwise(ir, rec)
            if partition_info[new_node].count > 1:
                # Collapse down to single partition
                inter = Repartition(new_node.schema, new_node)
                partition_info[inter] = PartitionInfo(count=1)
                # Sort reduced partition
                new_node = ir.reconstruct([inter])
                partition_info[new_node] = PartitionInfo(count=1)
            return new_node, partition_info

    # TODO: Add general multi-partition Sort support
    return _lower_ir_fallback(
        ir, rec, msg=f"Class {type(ir)} does not support multiple partitions."
    )
