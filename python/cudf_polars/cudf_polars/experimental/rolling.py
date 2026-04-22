# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rolling window lowering for the parallel/streaming engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import Rolling, Slice
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.utils import _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.experimental.dispatch import LowerIRTransformer


@lower_ir_node.register(Rolling)
def _(
    ir: Rolling, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    config_options = rec.state["config_options"]

    if (
        config_options.executor.runtime != "rapidsmpf"
        or config_options.executor.cluster not in ("single", "spmd")
        or (
            config_options.executor.cluster != "spmd"
            and (spmd_ctx := config_options.executor.spmd_context) is not None
            and spmd_ctx.comm.nranks > 1
        )
    ):
        # Single-partition fallback
        return _lower_ir_fallback(
            ir,
            rec,
            msg="Rolling with multiple partitions requires the rapidsmpf streaming backend.",
        )
    elif ir.zlice is not None:
        inner_rolling = Rolling(
            ir.schema,
            ir.index,
            ir.index_dtype,
            ir.preceding_ordinal,
            ir.following_ordinal,
            ir.closed_window,
            ir.keys,
            ir.agg_requests,
            None,
            ir.children[0],
        )
        return rec(Slice(ir.schema, *ir.zlice, inner_rolling))

    child, partition_info = rec(ir.children[0])
    new_rolling = ir.reconstruct([child])
    partition_info[new_rolling] = partition_info[child]
    return new_rolling, partition_info
