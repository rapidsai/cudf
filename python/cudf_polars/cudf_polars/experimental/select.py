# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Select Logic."""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import Select
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node

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


@lower_ir_node.register(Select)
def _(
    ir: Select, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)
    new_node = ir.reconstruct(children)

    # Search the expression graph for "complex" operations
    for ne in ir.exprs:
        for expr in traversal(ne.value):
            if type(expr).__name__ not in _PARTWISE:
                # TODO: Lower Scan node into something else if an
                # aggregattion is required. We assume that anything
                # left as a Scan is partitionwise or single-partition.
                if any(partition_info[c].count > 1 for c in children):
                    raise NotImplementedError(
                        f"Expr {type(expr)} does not support multiple partitions."
                    )
                else:  # pragma: no cover
                    partition_info[new_node] = PartitionInfo(count=1)
                    return new_node, partition_info

    # Remaining Select ops are partition-wise
    partition_info[new_node] = PartitionInfo(
        count=max(partition_info[c].count for c in children)
    )
    return new_node, partition_info
