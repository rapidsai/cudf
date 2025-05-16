# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sorting Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import Sort
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.utils import _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.dispatch import LowerIRTransformer


@lower_ir_node.register(Sort)
def _(
    ir: Sort, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Special handling for slicing
    # (May be a top- or bottom-k operation)

    if ir.zlice is not None and ir.zlice[0] < 1:
        # TODO: Handle large slices (e.g. 1m+ rows)
        from cudf_polars.experimental.parallel import _lower_ir_pwise

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

    # Fallback
    return _lower_ir_fallback(ir, rec, msg="Sort does not support multiple partitions.")
