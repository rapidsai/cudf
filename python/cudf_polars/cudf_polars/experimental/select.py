# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Select Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import Select
from cudf_polars.experimental.parallel import (
    PartitionInfo,
    _default_lower_ir_node,
    _lower_children,
    _partitionwise_ir_tasks,
    generate_ir_tasks,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


_PARTWISE = (
    "Literal",
    "LiteralColumn",
    "Col",
    "ColRef",
    "BooleanFunction",
    "StringFunction",
    "TemporalFunction",
    "Filter",
    "Cast",
    "Ternary",
    "BinOp",
    "UnaryFunction",
)


class PartwiseSelect(Select):
    """Partitionwise Select operation."""


def lower_select_node(
    ir: Select, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Rewrite a GroupBy node with proper partitioning."""
    from cudf_polars.dsl.traversal import traversal

    # Lower children first
    children, partition_info = _lower_children(ir, rec)

    # Search the expressions for "complex" operations
    for ne in ir.exprs:
        for expr in traversal(ne.value):
            if type(expr).__name__ not in _PARTWISE:
                return _default_lower_ir_node(ir, rec)

    # Remaining Select ops are partition-wise
    new_node = PartwiseSelect(
        ir.schema,
        ir.exprs,
        ir.should_broadcast,
        *children,
    )
    partition_info[new_node] = PartitionInfo(
        count=max(partition_info[c].count for c in children)
    )
    return new_node, partition_info


@generate_ir_tasks.register(PartwiseSelect)
def _(
    ir: PartwiseSelect, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    return _partitionwise_ir_tasks(ir, partition_info)
