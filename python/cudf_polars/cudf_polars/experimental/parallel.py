# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Partitioned LogicalPlan nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR


class PartitionInfo:
    """
    Partitioning information.

    This class only tracks the partition count (for now).
    """

    __slots__ = ("npartitions",)

    def __init__(self, npartitions: int):
        self.npartitions = npartitions


@runtime_checkable
class PartitionedIR(Protocol):
    """
    Partitioned IR Protocol.

    IR nodes must satistfy this protocol to generate a valid task graph.
    """

    _key: str
    _parts: PartitionInfo

    def _tasks(self) -> MutableMapping:
        raise NotImplementedError()


def task_graph(_ir: IR) -> tuple[MutableMapping[str, Any], str]:
    """Construct a Dask-compatible task graph."""
    from cudf_polars.dsl.traversal import traversal
    from cudf_polars.experimental.single import lower_ir_graph

    # Rewrite IR graph into a ParIR graph
    ir: PartitionedIR = lower_ir_graph(_ir)

    dsk = {
        k: v for layer in [n._tasks() for n in traversal(ir)] for k, v in layer.items()
    }

    # Add task to reduce output partitions
    npartitions = ir._parts.npartitions
    key_name = ir._key
    if npartitions == 1:
        dsk[key_name] = (key_name, 0)
    else:
        # Need DataFrame.concat support
        raise NotImplementedError()

    return dsk, key_name


def evaluate_dask(ir: IR) -> DataFrame:
    """Evaluate an IR graph with Dask."""
    from dask import get

    dsk, key = task_graph(ir)
    return get(dsk, key)
