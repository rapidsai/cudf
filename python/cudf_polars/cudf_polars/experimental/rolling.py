# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rolling window lowering for the parallel/streaming engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl.ir import Rolling, Sort
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.utils import _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.experimental.dispatch import LowerIRTransformer


def _child_sort_matches_rolling_stream_spec(child: IR, ir: Rolling) -> bool:
    """True if ``child`` is already a Sort with the same key order rapidsmpf would insert."""
    if not isinstance(child, Sort):
        return False
    need = (*ir.keys, ir.index)
    if len(child.by) != len(need):
        return False
    if any(a.name != b.name for a, b in zip(child.by, need, strict=True)):
        return False
    if any(o != plc.types.Order.ASCENDING for o in child.order):
        return False
    if any(n != plc.types.NullOrder.BEFORE for n in child.null_order):
        return False
    return child.zlice is None


@lower_ir_node.register(Rolling)
def _(
    ir: Rolling, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    config_options = rec.state["config_options"]

    if config_options.executor.runtime != "rapidsmpf":
        return _lower_ir_fallback(
            ir,
            rec,
            msg="Rolling with multiple partitions requires the rapidsmpf streaming backend.",
        )

    # Polars guarantees the window column is sorted ascending (ungrouped) or sorted
    # within each group (grouped); we do not add Sort in dsl/utils/rolling.py for that.
    # Make sure input data is sorted
    child = ir.children[0]
    if _child_sort_matches_rolling_stream_spec(child, ir):
        sorted_child, partition_info = rec(child)
    else:
        # TODO(rolling): For grouped rolling, consider hash-shuffle on keys + local sort
        # (index within partition) instead of a global sort.
        sort_by = (*ir.keys, ir.index)
        sort_order = tuple(plc.types.Order.ASCENDING for _ in sort_by)
        sort_null_order = tuple(plc.types.NullOrder.BEFORE for _ in sort_by)
        sort_node = Sort(
            child.schema,
            sort_by,
            sort_order,
            sort_null_order,
            stable=True,
            zlice=None,
            df=child,
        )
        sorted_child, partition_info = rec(sort_node)

    new_rolling = ir.reconstruct([sorted_child])
    partition_info[new_rolling] = partition_info[sorted_child]
    return new_rolling, partition_info
