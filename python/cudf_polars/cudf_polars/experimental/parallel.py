# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Partitioned LogicalPlan nodes."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.nodebase import PartitionInfo
from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.nodebase import Node


def get_key_name(node: Node) -> str:
    """Generate the key name for a Node."""
    return f"{type(node).__name__.lower()}{hash(node)}"


@singledispatch
def partition_info_dispatch(node: Node) -> PartitionInfo:
    """Return partitioning information for a given node."""
    # Assume the partition count is preserved by default.
    count = 1
    if node.children:
        count = max(child.parts.count for child in node.children)
    if count > 1:
        raise NotImplementedError(
            f"Multi-partition support is not implemented for {type(node)}."
        )
    return PartitionInfo(count=count)


@singledispatch
def generate_tasks(ir: IR) -> MutableMapping[Any, Any]:
    """Generate tasks for an IR node."""
    if ir.parts.count == 1:
        key_name = get_key_name(ir)
        return {
            (key_name, 0): (
                ir.do_evaluate,
                *ir._non_child_args,
                *((get_key_name(child), 0) for child in ir.children),
            )
        }
    raise NotImplementedError(f"Cannot generate tasks for {ir}.")


def task_graph(ir: IR) -> tuple[MutableMapping[str, Any], str]:
    """Construct a Dask-compatible task graph."""
    # NOTE: It may be necessary to add an optimization
    # pass here to "rewrite" the single-partition IR graph.

    dsk = {
        k: v
        for layer in [generate_tasks(n) for n in traversal(ir)]
        for k, v in layer.items()
    }

    # Add task to reduce output partitions
    key_name = get_key_name(ir)
    if ir.parts.count == 1:
        dsk[key_name] = (key_name, 0)
    else:
        # Need DataFrame.concat support
        raise NotImplementedError("Multi-partition output is not supported.")

    return dsk, key_name


def evaluate_dask(ir: IR) -> DataFrame:
    """Evaluate an IR graph with Dask."""
    from dask import get

    dsk, key = task_graph(ir)
    return get(dsk, key)
