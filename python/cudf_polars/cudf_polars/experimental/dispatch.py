# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition dispatch functions."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from typing import TypeAlias

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.typing import GenericTransformer


LowerIRTransformer: TypeAlias = (
    "GenericTransformer[IR, tuple[IR, MutableMapping[IR, PartitionInfo]]]"
)
"""Protocol for Lowering IR nodes."""


@singledispatch
def lower_ir_node(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR node and extract partitioning information.

    Parameters
    ----------
    ir
        IR node to rewrite.
    rec
        Recursive LowerIRTransformer callable.

    Returns
    -------
    new_ir, partition_info
        The rewritten node, and a mapping from unique nodes in
        the full IR graph to associated partitioning information.

    Notes
    -----
    This function is used by `lower_ir_graph`.

    See Also
    --------
    lower_ir_graph
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover


@singledispatch
def generate_ir_tasks(
    ir: IR, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    """
    Generate a task graph for evaluation of an IR node.

    Parameters
    ----------
    ir
        IR node to generate tasks for.
    partition_info
        Partitioning information, obtained from :func:`lower_ir_graph`.

    Returns
    -------
    mapping
        A (partial) dask task graph for the evaluation of an ir node.

    Notes
    -----
    Task generation should only produce the tasks for the current node,
    referring to child tasks by name.

    See Also
    --------
    task_graph
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover
