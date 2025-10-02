# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core lowering logic for the RapidsMPF streaming engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cudf_polars.experimental.rapidsmpf.io  # noqa: F401
from cudf_polars.dsl.ir import (
    IR,
    Cache,
    Filter,
    HConcat,
    MapFunction,
    Projection,
    Slice,
    Union,
)
from cudf_polars.experimental.rapidsmpf.dispatch import (
    lower_ir_node,
)
from cudf_polars.experimental.utils import _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.experimental.rapidsmpf.dispatch import LowerIRTransformer


@lower_ir_node.register(IR)
def _(ir: IR, rec: LowerIRTransformer) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Default logic - Fall back to a single partition/chunk.
    return _lower_ir_fallback(
        ir, rec, msg=f"Class {type(ir)} does not support multiple partitions."
    )


@lower_ir_node.register(Projection)
@lower_ir_node.register(Cache)
@lower_ir_node.register(HConcat)
@lower_ir_node.register(Slice)
@lower_ir_node.register(Filter)
@lower_ir_node.register(MapFunction)
@lower_ir_node.register(Union)
def _lower_ir_node_task_engine(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Use task-engine lowering logic
    from cudf_polars.experimental.dispatch import lower_ir_node as base_lower_ir_node

    return base_lower_ir_node(ir, rec)
