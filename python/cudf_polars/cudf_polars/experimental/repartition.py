# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Repartitioning Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import IR
from cudf_polars.experimental.base import get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.experimental.parallel import PartitionInfo
    from cudf_polars.typing import Schema


class Repartition(IR):
    """Repartition a DataFrame."""

    __slots__ = ()
    _non_child = ("schema",)

    def __init__(self, schema: Schema, df: IR):
        self.schema = schema
        self._non_child_args = ()
        self.children = (df,)


@generate_ir_tasks.register(Repartition)
def _(
    ir: Repartition, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    # Repartition an IR node.
    # Only supports N -> 1 rapartitioning (for now).

    if partition_info[ir].count > 1:  # pragma: no cover
        raise NotImplementedError(
            f"Repartition -> {partition_info[ir].count} not supported."
        )

    (child,) = ir.children
    key_name = get_key_name(ir)
    child_name = get_key_name(child)
    return {
        (key_name, 0): (
            _concat,
            *[(child_name, idx) for idx in range(partition_info[child].count)],
        )
    }
