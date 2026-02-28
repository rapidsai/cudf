# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
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
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.shuffle import Shuffle
from cudf_polars.experimental.utils import (
    _dynamic_planning_on,
    _fallback_inform,
    _get_unique_fractions,
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


def lower_distinct(
    ir: Distinct,
    child: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    *,
    unique_fraction: float | None = None,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Lower a Distinct IR into partition-wise stages.

    Note: Edge cases (KEEP_NONE + ordering, complex slice, pre-shuffle)
    must be handled by the caller before calling this function.

    Parameters
    ----------
    ir
        The Distinct IR node to lower.
    child
        The reconstructed child of ``ir``. May differ from ``ir.children[0]``.
        If KEEP_NONE, caller must ensure child is already shuffled on keys.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.
    unique_fraction
        Fraction of unique values to total values. Used for algorithm selection.
        A value of `1.0` means the column is unique.

    Returns
    -------
    new_node
        The lowered Distinct node.
    partition_info
        A mapping from unique nodes in the new graph to associated
        partitioning information.
    """
    subset: frozenset[str] = ir.subset or frozenset(ir.schema)
    distinct_keys = tuple(
        NamedExpr(name, Col(ir.schema[name], name)) for name in subset
    )

    child_count = partition_info[child].count
    shuffled = partition_info[child].partitioned_on == distinct_keys

    # Check for ordering requirements (shuffle is not stable)
    require_tree_reduction = ir.stable or ir.keep in (
        plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
    )

    output_count = 1
    n_ary = 32  # Arbitrary default (for now)
    if ir.zlice is not None and ir.zlice[1] is not None:
        # Head/tail slice operation has been pushed into Distinct
        # (caller ensures only simple slices reach here)
        n_ary = max(1_000_000 // ir.zlice[1], 2)
    elif unique_fraction is not None:
        # Use unique_fraction to determine partitioning
        n_ary = min(max(int(1.0 / unique_fraction), 2), child_count)
        output_count = max(int(unique_fraction * child_count), 1)

    if output_count > 1 and require_tree_reduction:
        # Need to reduce down to a single partition even
        # if the unique_fraction is large.
        output_count = 1
        _fallback_inform(
            "Unsupported unique options for multiple partitions.",
            config_options,
        )

    # Partition-wise unique
    count = child_count
    new_node: IR = ir.reconstruct([child])
    partition_info[new_node] = PartitionInfo(count=count)

    if shuffled or output_count == 1:
        # Tree reduction
        while count > output_count:
            new_node = Repartition(new_node.schema, new_node)
            count = max(math.ceil(count / n_ary), output_count)
            partition_info[new_node] = PartitionInfo(count=count)
            new_node = ir.reconstruct([new_node])
            partition_info[new_node] = PartitionInfo(count=count)
    else:
        # Shuffle
        new_node = Shuffle(
            new_node.schema,
            distinct_keys,
            config_options.executor.shuffle_method,
            config_options.executor.shuffler_insertion_method,
            new_node,
        )
        partition_info[new_node] = PartitionInfo(count=output_count)
        new_node = ir.reconstruct([new_node])
        partition_info[new_node] = PartitionInfo(
            count=output_count,
            partitioned_on=distinct_keys,
        )

    return new_node, partition_info


@lower_ir_node.register(Distinct)
def _(
    ir: Distinct, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Extract child partitioning
    original_child = ir.children[0]
    child, partition_info = rec(ir.children[0])
    child_count = partition_info[child].count
    subset: frozenset[str] = ir.subset or frozenset(ir.schema)
    distinct_keys = tuple(
        NamedExpr(name, Col(ir.schema[name], name)) for name in subset
    )

    config_options = rec.state["config_options"]

    # Check for ordering requirements (shuffle is not stable)
    require_tree = ir.stable or ir.keep in (
        plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
    )

    # Handle edge cases upfront
    # KEEP_NONE + ordering = unsupported (need shuffle but shuffle is unstable)
    if ir.keep == plc.stream_compaction.DuplicateKeepOption.KEEP_NONE:
        if require_tree:
            return _lower_ir_fallback(
                ir,
                rec,
                msg="Unsupported unique options for multiple partitions.",
            )

        # KEEP_NONE needs pre-shuffle (must see all duplicates to drop them)
        if partition_info[child].partitioned_on != distinct_keys:
            child = Shuffle(
                child.schema,
                distinct_keys,
                config_options.executor.shuffle_method,
                config_options.executor.shuffler_insertion_method,
                child,
            )
            partition_info[child] = PartitionInfo(
                count=child_count,
                partitioned_on=distinct_keys,
            )

    # Complex slice = unsupported (offset >= 1 or no length)
    if ir.zlice is not None and (ir.zlice[0] >= 1 or ir.zlice[1] is None):
        return _lower_ir_fallback(
            ir,
            rec,
            msg="Complex slice not supported for multiple partitions.",
        )

    # Branch based on dynamic planning
    if _dynamic_planning_on(
        config_options
    ):  # pragma: no cover; Requires rapidsmpf runtime
        # Dynamic planning: Reconstruct the Distinct.
        # The runtime distinct_node will handle strategy selection,
        # respecting ir.stable, ir.keep, and ir.zlice attributes.
        dynamic_node = ir.reconstruct([child])
        partition_info[dynamic_node] = PartitionInfo(
            count=child_count,
            partitioned_on=distinct_keys,
        )
        return dynamic_node, partition_info

    # Non-dynamic planning: use unique_fraction heuristics
    unique_fraction_dict = _get_unique_fractions(
        tuple(subset),
        config_options.executor.unique_fraction,
        row_count=rec.state["stats"].row_count.get(original_child),
        column_stats=rec.state["stats"].column_stats.get(original_child),
    )
    unique_fraction = (
        max(unique_fraction_dict.values()) if unique_fraction_dict else None
    )

    return lower_distinct(
        ir,
        child,
        partition_info,
        config_options,
        unique_fraction=unique_fraction,
    )
