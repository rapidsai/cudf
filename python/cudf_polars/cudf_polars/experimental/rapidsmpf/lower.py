# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core lowering logic for the RapidsMPF streaming engine."""

from __future__ import annotations

import operator
from functools import partial, reduce
from typing import TYPE_CHECKING

import cudf_polars.experimental.rapidsmpf.io  # noqa: F401
from cudf_polars.dsl.ir import (
    IR,
    Cache,
    HConcat,
    Projection,
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


def _lower_ir_pwise(
    ir: IR, rec: LowerIRTransformer, *, preserve_partitioning: bool = False
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Lower a partition-wise (i.e. embarrassingly-parallel) IR node

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)
    counts = {partition_info[c].count for c in children}

    # Check that child partitioning is supported
    if len(counts) > 1:  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg=f"Class {type(ir)} does not support children with mismatched partition counts.",
        )

    # Preserve child partition_info if possible
    if preserve_partitioning and len(children) == 1:
        partition = partition_info[children[0]]
    else:
        partition = PartitionInfo(count=max(counts))

    # Return reconstructed node and partition-info dict
    new_node = ir.reconstruct(children)
    partition_info[new_node] = partition
    return new_node, partition_info


_lower_ir_pwise_preserve = partial(_lower_ir_pwise, preserve_partitioning=True)
lower_ir_node.register(Projection, _lower_ir_pwise_preserve)
lower_ir_node.register(Cache, _lower_ir_pwise)
lower_ir_node.register(HConcat, _lower_ir_pwise)


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
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_sub_network'"
    )
    if (
        config_options.executor.scheduler == "distributed"
    ):  # pragma: no cover; Requires distributed
        raise NotImplementedError(
            "Fallback is not yet supported distributed execution "
            "with the RAPIDS-MPF streaming engine."
        )

    # Lower children
    lowered_children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Ensure all children are single-partitioned
    children = []
    inform = False
    for c in lowered_children:
        if partition_info[c].count > 1:
            # Fall-back logic
            inform = True
        # Always use a Rechunk node to ensure a single chunk is produced.
        # The Rechunk node will be a no-op if the child only produces
        # a single chunk at run-time, but we don't know the chunk
        # count ahead of time.
        child = Rechunk(
            c.schema,
            "chunk_count",
            1,
            c,
        )
        partition_info[child] = PartitionInfo(count=1)
        children.append(child)

    if inform and msg:
        # Warn/raise the user if "fallback_mode" is not "silent"
        _fallback_inform(msg, config_options)

    # Reconstruct and return
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=1)
    return new_node, partition_info
