# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core lowering logic for the RAPIDS-MPF streaming engine."""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

import cudf_polars.experimental.rapidsmpf.io  # noqa: F401
from cudf_polars.dsl.ir import (
    IR,
    Cache,
    Filter,
    HConcat,
    HStack,
    MapFunction,
    Projection,
    Union,
)
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.rapidsmpf.dispatch import (
    lower_ir_node,
)
from cudf_polars.experimental.rapidsmpf.rechunk import Rechunk

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.experimental.rapidsmpf.dispatch import LowerIRTransformer


@lower_ir_node.register(IR)
def _(ir: IR, rec: LowerIRTransformer) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Default logic - Fall back to a single partition/chunk.
    return _lower_ir_fallback(
        ir, rec, msg=f"Class {type(ir)} does not support multiple partitions."
    )


@lower_ir_node.register(Union)
@lower_ir_node.register(MapFunction)
@lower_ir_node.register(Projection)
@lower_ir_node.register(Cache)
@lower_ir_node.register(HConcat)
@lower_ir_node.register(Filter)
@lower_ir_node.register(HStack)
def _lower_ir_simple(
    ir: IR,
    rec: LowerIRTransformer,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Simple single-partition lowering logic.
    # Use the same lowering logic as the task-based engine.
    from cudf_polars.experimental.dispatch import lower_ir_node as base_lower_ir_node

    return base_lower_ir_node(ir, rec)


def _lower_ir_fallback(
    ir: IR,
    rec: LowerIRTransformer,
    *,
    msg: str | None = None,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Catch-all single-partition lowering logic.
    # If any children contain multiple partitions,
    # those children will be collapsed with `Rechunk`.
    from cudf_polars.experimental.utils import _fallback_inform

    # NOTE: This logic is largely a copy-and-paste of
    # `cudf_polars.experimental.utils._lower_ir_fallback`.
    # We use a separate function for now since the logic
    # is likely to fluctuate while the streaming engine
    # is under heavy development.

    # NOTE: (IMPORTANT) Since Rechunk is a local operation,
    # the current fallback logic will only work for one rank!
    ctx = rec.state["ctx"]
    if ctx.comm().nranks > 1:  # pragma: no cover; Requires multiple ranks
        raise NotImplementedError(
            "Fallback is not yet supported multi-rank execution "
            "with the RAPIDS-MPF streaming engine."
        )

    # Lower children
    lowered_children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Ensure all children are single-partitioned
    children = []
    fallback = False
    for c in lowered_children:
        child = c
        if partition_info[c].count > 1:
            # Fall-back logic
            fallback = True
            child = Rechunk(
                child.schema,
                "single",
                child,
            )
            partition_info[child] = PartitionInfo(count=1)
        children.append(child)

    if fallback and msg:
        # Warn/raise the user if any children were collapsed
        # and the "fallback_mode" configuration is not "silent"
        _fallback_inform(msg, rec.state["config_options"])

    # Reconstruct and return
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=1)
    return new_node, partition_info
