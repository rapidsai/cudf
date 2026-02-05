# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Repartitioning Logic."""

from __future__ import annotations

import itertools
from functools import partial
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import IR
from cudf_polars.experimental.base import get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.parallel import PartitionInfo
    from cudf_polars.typing import Schema


class Repartition(IR):
    """
    Repartition a DataFrame.

    Notes
    -----
    Repartitioning means that we are not modifying any
    data, nor are we reordering or shuffling rows. We
    are only changing the overall partition count. For
    now, we only support an N -> [1...N] repartitioning
    (inclusive). The output partition count is tracked
    separately using PartitionInfo.
    """

    __slots__ = ()
    _non_child = ("schema",)
    _n_non_child_args = 0

    def __init__(self, schema: Schema, df: IR):
        self.schema = schema
        self._non_child_args = ()
        self.children = (df,)


@generate_ir_tasks.register(Repartition)
def _(
    ir: Repartition,
    partition_info: MutableMapping[IR, PartitionInfo],
    context: IRExecutionContext,
) -> MutableMapping[Any, Any]:
    # Repartition an IR node.
    # Only supports rapartitioning to fewer (for now).

    (child,) = ir.children
    count_in = partition_info[child].count
    count_out = partition_info[ir].count

    if count_out > count_in:  # pragma: no cover
        raise NotImplementedError(
            f"Repartition {count_in} -> {count_out} not supported."
        )

    key_name = get_key_name(ir)
    n, remainder = divmod(count_in, count_out)
    # Spread remainder evenly over the partitions.
    offsets = [0, *itertools.accumulate(n + (i < remainder) for i in range(count_out))]
    child_keys = tuple(partition_info[child].keys(child))
    return {
        (key_name, i): (
            partial(_concat, context=context),
            *child_keys[offsets[i] : offsets[i + 1]],
        )
        for i in range(count_out)
    }
