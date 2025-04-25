# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition Distinct logic."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl.expressions.base import Col, NamedExpr
from cudf_polars.dsl.ir import Distinct
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.utils import _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.dispatch import LowerIRTransformer


@lower_ir_node.register(Distinct)
def _(
    ir: Distinct, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Extract child partitioning
    child, partition_info = rec(ir.children[0])
    child_count = partition_info[child].count
    config_options = rec.state["config_options"]
    require_tree_reduction = ir.stable or ir.keep in (
        plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
    )

    subset: frozenset = ir.subset or frozenset(ir.schema)
    if ir.zlice is not None:
        # Head/tail slice operation has been pushed into Distinct
        if ir.zlice[0] < 1 and ir.zlice[1] is not None:
            # Use rough 1m-row heuristic to set n_ary
            n_ary = max(int(1_000_000 / ir.zlice[1]), 2)
            output_count = 1
        else:
            # Slice is not head or tail - Can this happen?
            return _lower_ir_fallback(
                ir, rec, msg="Unsupported slice for multiple partitions."
            )
    else:
        # No head/tail slice - Use cardinality to determine partitioning
        cardinality_factor = {
            c: min(f, 1.0)
            for c, f in config_options.executor.cardinality_factor.items()
            if c in subset
        }
        if cardinality_factor:
            cardinality = max(cardinality_factor.values())
            n_ary = max(int(1.0 / cardinality), 2)
            output_count = max(int(cardinality * child_count), 1)
        else:
            output_count = 1
            n_ary = 32  # Arbitrary default (for now)

    if output_count > 1 and require_tree_reduction:
        # TODO: Use fallback config to warn or raise?
        output_count = 1

    new_node: IR
    if output_count > 1:
        # Distinct -> Shuffle -> Distinct
        from cudf_polars.experimental.shuffle import Shuffle

        new_node = ir.reconstruct([child])
        partition_info[new_node] = PartitionInfo(count=child_count)
        shuffle_keys = tuple(
            NamedExpr(name, Col(ir.schema[name], name)) for name in subset
        )
        new_node = Shuffle(new_node.schema, shuffle_keys, config_options, new_node)
        partition_info[new_node] = PartitionInfo(count=output_count)
        new_node = ir.reconstruct([new_node])
        partition_info[new_node] = PartitionInfo(
            count=output_count,
            partitioned_on=shuffle_keys,
        )
    else:
        # Tree reduction
        from cudf_polars.experimental.repartition import Repartition

        count = child_count
        new_node = ir.reconstruct([child])
        partition_info[new_node] = PartitionInfo(count=count)
        while count > 1:
            new_node = Repartition(new_node.schema, new_node)
            count = max(math.ceil(count / n_ary), 1)
            partition_info[new_node] = PartitionInfo(count=count)
            new_node = ir.reconstruct([new_node])
            partition_info[new_node] = PartitionInfo(count=count)

    return new_node, partition_info
