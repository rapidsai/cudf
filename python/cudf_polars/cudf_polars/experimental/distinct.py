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
from cudf_polars.experimental.utils import (
    _fallback_inform,
    _get_unique_fractions,
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions


def lower_distinct(
    ir: Distinct,
    child: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    *,
    unique_fraction: float | None = None,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Lower a Distinct IR into partition-wise stages.

    Parameters
    ----------
    ir
        The Distinct IR node to lower.
    child
        The reconstructed child of ``ir``. May differ
        from ``ir.children[0]``.
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
    from cudf_polars.experimental.repartition import Repartition
    from cudf_polars.experimental.shuffle import Shuffle

    # Extract child partitioning
    child_count = partition_info[child].count
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'lower_distinct'"
    )

    # Assume shuffle is not stable for now. Therefore, we
    # require a tree reduction if row order matters.
    require_tree_reduction = ir.stable or ir.keep in (
        plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
    )

    subset: frozenset = ir.subset or frozenset(ir.schema)
    shuffle_keys = tuple(NamedExpr(name, Col(ir.schema[name], name)) for name in subset)
    shuffled = partition_info[child].partitioned_on == shuffle_keys
    if ir.keep == plc.stream_compaction.DuplicateKeepOption.KEEP_NONE:
        # Need to shuffle the original data for keep == "none"
        if require_tree_reduction:
            # TODO: We cannot drop all duplicates without
            # shuffling the data up front, and we assume
            # shuffling is unstable for now. Note that the
            # task-based shuffle should be stable, but it
            # its performance is very poor.
            raise NotImplementedError(
                "Unsupported unique options for multiple partitions."
            )
        if not shuffled:
            child = Shuffle(
                child.schema,
                shuffle_keys,
                config_options.executor.shuffle_method,
                child,
            )
            partition_info[child] = PartitionInfo(
                count=child_count,
                partitioned_on=shuffle_keys,
            )
            shuffled = True

    output_count = 1
    n_ary = 32  # Arbitrary default (for now)
    if ir.zlice is not None:
        # Head/tail slice operation has been pushed into Distinct
        if ir.zlice[0] < 1 and ir.zlice[1] is not None:
            # Use rough 1m-row heuristic to set n_ary
            n_ary = max(int(1_000_000 / ir.zlice[1]), 2)
        else:  # pragma: no cover
            # TODO: General slicing is not supported for multiple
            # partitions. For now, we raise an error to fall back
            # to one partition.
            raise NotImplementedError("Unsupported slice for multiple partitions.")
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
            shuffle_keys,
            config_options.executor.shuffle_method,
            new_node,
        )
        partition_info[new_node] = PartitionInfo(count=output_count)
        new_node = ir.reconstruct([new_node])
        partition_info[new_node] = PartitionInfo(
            count=output_count,
            partitioned_on=shuffle_keys,
        )

    return new_node, partition_info


@lower_ir_node.register(Distinct)
def _(
    ir: Distinct, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Extract child partitioning
    original_child = ir.children[0]
    child, partition_info = rec(ir.children[0])
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node'"
    )

    subset: frozenset[str] = ir.subset or frozenset(ir.schema)
    unique_fraction_dict = _get_unique_fractions(
        tuple(subset),
        config_options.executor.unique_fraction,
        row_count=rec.state["stats"].row_count.get(original_child),
        column_stats=rec.state["stats"].column_stats.get(original_child),
    )
    unique_fraction = (
        max(unique_fraction_dict.values()) if unique_fraction_dict else None
    )

    try:
        return lower_distinct(
            ir,
            child,
            partition_info,
            config_options,
            unique_fraction=unique_fraction,
        )
    except NotImplementedError as err:
        return _lower_ir_fallback(ir, rec, msg=str(err))
