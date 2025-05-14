# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""IR node fusion."""

from __future__ import annotations

import operator
from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import (
    IR,
    Cache,
    Filter,
    GroupBy,
    HStack,
    MapFunction,
    Projection,
    Select,
    Sort,
    Union,
)
from cudf_polars.dsl.traversal import CachingVisitor, traversal
from cudf_polars.experimental.dispatch import generate_ir_tasks
from cudf_polars.experimental.io import Scan, SplitScan

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping, Sequence
    from typing import Any

    from cudf_polars.containers import DataFrame
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions


class Fused(IR):
    """Fused IR node."""

    __slots__ = ("fused_io", "subnodes")
    _non_child = ("schema", "subnodes", "fused_io")
    subnodes: tuple[IR, ...]
    """Fused sub-nodes."""
    fused_io: Union | None
    """Fused IO node."""

    def __init__(
        self,
        schema: Schema,
        subnodes: tuple[IR, ...],
        fused_io: Union | None,
        *children: IR,
    ):
        self.schema = schema
        self.subnodes = subnodes
        self.fused_io = fused_io
        self._non_child_args = (
            [node.do_evaluate for node in subnodes],
            [tuple(node._non_child_args) for node in self.subnodes],
            [],
            [],
        )
        self.children = children

    @classmethod
    def do_evaluate(
        cls,
        funcs: Sequence[Callable],
        subargs: Sequence[Sequence[Any]],
        io_funcs: Sequence[Callable],
        io_args: Sequence[Any],
        *children: DataFrame,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        if io_funcs:
            children = (io_funcs[0](*io_args),)
        for func, args in zip(funcs, subargs, strict=False):
            children = (func(*args, *children),)
        return children[0]


def _fuse_ir_node(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    partition_info: MutableMapping[IR, PartitionInfo] = {}
    children = ()
    if ir.children:
        children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
        partition_info = reduce(operator.or_, _partition_info)

    new_node: IR
    if ir in rec.state["fusable"]:
        if isinstance(ir, Union):
            new_node = Fused(ir.schema, (), ir)
        elif len(children) == 1 and isinstance(children[0], Fused):
            (child,) = children
            grandchildren = child.children
            new_node = Fused(
                ir.schema, (*child.subnodes, ir), child.fused_io, *grandchildren
            )
        else:
            new_node = Fused(ir.schema, (ir,), None, *children)
    else:
        new_node = ir.reconstruct(children)

    partition_info[new_node] = rec.state["partition_info"][ir]
    return new_node, partition_info


def _fusable_ir_type(ir: IR) -> bool:
    return (
        # Basic fusion
        isinstance(
            ir, (Cache, Filter, GroupBy, HStack, MapFunction, Projection, Select, Sort)
        )
        or (
            # IO Fusion
            isinstance(ir, Union)
            and all(isinstance(n, (Scan, SplitScan)) for n in ir.children)
        )
    )


@generate_ir_tasks.register(Fused)
def _(
    ir: Fused, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    from cudf_polars.experimental.parallel import _default_generate_ir_tasks

    if ir.fused_io is None:
        return _default_generate_ir_tasks(ir, partition_info)

    assert isinstance(ir.fused_io, Union)
    graph: MutableMapping[Any, Any] = {}
    for i, key in enumerate(partition_info[ir].keys(ir)):
        io = ir.fused_io.children[i]
        assert isinstance(io, (Scan, SplitScan))
        graph[key] = (
            ir.do_evaluate,
            ir._non_child_args[0],
            ir._non_child_args[1],
            [io.do_evaluate],
            list(io._non_child_args),
        )
    return graph


def fuse_ir_graph(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR graph with fused nodes.

    Parameters
    ----------
    ir
        Root of the graph to rewrite.
    partition_info
        Initial partitioning information.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    new_ir, partition_info
        The rewritten graph, and a mapping from unique nodes
        in the new graph to associated partitioning information.
    """
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'fuse_ir_graph'"
    )
    if not config_options.executor.task_fusion:
        return ir, partition_info

    parents: defaultdict[IR, int] = defaultdict(int)
    if _fusable_ir_type(ir):
        parents[ir] = 1
    for node in traversal([ir]):
        for child in node.children:
            if _fusable_ir_type(child):
                # Basic fusion
                parents[child] += 1
    fusable = {node for node, count in parents.items() if count == 1}
    state = {"fusable": fusable, "partition_info": partition_info}
    mapper = CachingVisitor(_fuse_ir_node, state=state)
    return mapper(ir)
