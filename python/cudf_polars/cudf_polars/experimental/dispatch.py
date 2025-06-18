# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition dispatch functions."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl import ir
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.utils.config import ConfigOptions


class State(TypedDict):
    """
    State used for lowering IR nodes.

    Parameters
    ----------
    config_options
        GPUEngine configuration options.
    """

    config_options: ConfigOptions


LowerIRTransformer: TypeAlias = GenericTransformer[
    "ir.IR", "tuple[ir.IR, MutableMapping[ir.IR, PartitionInfo]]", State
]
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
