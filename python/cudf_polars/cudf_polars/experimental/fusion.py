# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""IR node fusion."""

from __future__ import annotations

import operator
from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import (
    IR,
    Cache,
    Filter,
    GroupBy,
    HStack,
    MapFunction,
    Projection,
    Select,
    Union,
)
from cudf_polars.dsl.traversal import CachingVisitor, traversal
from cudf_polars.experimental.dispatch import generate_ir_tasks
from cudf_polars.experimental.io import Scan, SplitScan

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping, Sequence
    from typing import Any

    from typing_extensions import Self

    from cudf_polars.containers import DataFrame
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema


class Fused(IR):
    """Fused IR node."""

    __slots__ = ("fusible", "subnodes")
    _non_child = ("schema", "subnodes", "fusible")
    subnodes: tuple[IR, ...]
    """Fused sub-nodes."""
    fusible: bool
    """Whether a parent may fuse to this node."""

    def __init__(
        self,
        schema: Schema,
        subnodes: tuple[IR, ...],
        fusible: bool,  # noqa: FBT001
        *children: IR,
    ):
        self.schema = schema
        self.subnodes = subnodes
        self.fusible = fusible
        self._non_child_args = (
            [node.do_evaluate for node in subnodes],
            [tuple(node._non_child_args) for node in self.subnodes],
        )
        self.children = children

    def fuse_parent(self, parent: IR, *, fusible: bool = True) -> Self:
        """Return a new IR node with a fused parent."""
        assert self.fusible, f"Fused node {self} is not fusible!"
        return type(self)(
            parent.schema, (*self.subnodes, parent), fusible, *self.children
        )

    @classmethod
    def from_ir(cls, ir: IR, *, fusible: bool = True) -> Self:
        "Construct from a single IR node."
        return cls(ir.schema, (ir,), fusible, *ir.children)

    @classmethod
    def do_evaluate(
        cls,
        funcs: Sequence[Callable],
        subargs: Sequence[Sequence[Any]],
        *children: DataFrame,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        for func, args in zip(funcs, subargs, strict=True):
            children = (func(*args, *children),)
        return children[0]


class FusedIO(IR):
    """Fused-IO IR node."""

    __slots__ = ("fused_io", "fusible", "subnodes")
    _non_child = ("schema", "subnodes", "fused_io", "fusible")
    subnodes: tuple[IR, ...]
    """Fused sub-nodes."""
    fused_io: Union
    """Fused IO node."""
    fusible: bool
    """Whether a parent may fuse to this node."""

    def __init__(
        self,
        schema: Schema,
        subnodes: tuple[IR, ...],
        fused_io: Union,
        fusible: bool,  # noqa: FBT001
    ):
        self.schema = schema
        self.subnodes = subnodes
        self.fused_io = fused_io
        self.fusible = fusible
        self._non_child_args = (
            [node.do_evaluate for node in subnodes],
            [tuple(node._non_child_args) for node in self.subnodes],
        )
        self.children = ()

    def fuse_parent(self, parent: IR, *, fusible: bool = True) -> Self:
        """Return a new IR node with a fused parent."""
        assert self.fusible, f"FusedIO node {self} is not fusible!"
        return type(self)(
            parent.schema,
            (*self.subnodes, parent),
            self.fused_io,
            fusible,
        )

    @classmethod
    def from_ir(cls, ir: Union, *, fusible: bool = True) -> Self:
        "Construct from a single IR node."
        return cls(ir.schema, (), ir, fusible)

    @classmethod
    def do_evaluate(
        cls,
        funcs: Sequence[Callable],
        subargs: Sequence[Sequence[Any]],
        io_cls: type[Scan | SplitScan],
        io_args: Sequence[Any],
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        return Fused.do_evaluate(funcs, subargs, io_cls.do_evaluate(*io_args))


def _maybe_make_fused(
    ir: IR,
    children: Sequence[IR],
    partition_info: MutableMapping[IR, PartitionInfo],
    *,
    fusible: bool = True,
) -> IR:
    """Return a Fused node if fusion is allowed."""
    new_node: IR
    if (
        isinstance(ir, Union)
        and all(isinstance(n, (Scan, SplitScan)) for n in ir.children)
        and partition_info[ir].count == len(ir.children)
    ):
        # IO Fusion.
        # Multi-partition IO is always implemented as a Union
        # over Scan/SplitScan nodes. The number of children
        # must correspond to the number of partitions.
        new_node = FusedIO.from_ir(ir, fusible=fusible)
    elif isinstance(
        ir,
        (Cache, Filter, GroupBy, HStack, MapFunction, Projection, Select),
    ):
        # Basic fusion.
        # These are partition-wise operations if they exist
        # in the graph after lowering (the data-movement
        # has been encoded with Shuffle/Reduction/etc).
        if isinstance(children[0], (Fused, FusedIO)) and children[0].fusible:
            new_node = children[0].fuse_parent(ir, fusible=fusible)
        else:
            new_node = Fused.from_ir(ir, fusible=fusible).reconstruct(children)
    else:
        new_node = ir.reconstruct(children)

    return new_node


def _fuse_ir_node(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    partition_info: MutableMapping[IR, PartitionInfo] = {}
    children = ()
    if ir.children:
        children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
        partition_info = reduce(operator.or_, _partition_info)

    new_node = _maybe_make_fused(
        ir,
        children,
        rec.state["partition_info"],
        # The new Fused node may only be fused by its
        # parent if it corresponds to a unique sub-plan.
        fusible=ir in rec.state["unique_subplans"],
    )
    partition_info[new_node] = rec.state["partition_info"][ir]
    return new_node, partition_info


@generate_ir_tasks.register(FusedIO)
def _(
    ir: FusedIO, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    graph: MutableMapping[Any, Any] = {}
    for i, key in enumerate(partition_info[ir].keys(ir)):
        io = ir.fused_io.children[i]
        assert isinstance(io, (Scan, SplitScan))
        graph[key] = (
            ir.do_evaluate,
            *ir._non_child_args,
            type(io),
            list(io._non_child_args),
        )
    return graph


def fuse_ir_graph(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR graph with fused nodes.

    Parameters
    ----------
    ir
        Root of the lowered-IR graph to rewrite.
    partition_info
        Initial partitioning information.

    Returns
    -------
    new_ir, partition_info
        The rewritten graph, and a mapping from unique nodes
        in the new graph to associated partitioning information.
    """
    parents: defaultdict[IR, int] = defaultdict(int)
    # if _is_fusible_ir_node(ir, partition_info):
    parents[ir] = 1
    for node in traversal([ir]):
        for child in node.children:
            # if _is_fusible_ir_node(child, partition_info):
            parents[child] += 1
    unique_subplans = {node for node, count in parents.items() if count == 1}
    state = {"unique_subplans": unique_subplans, "partition_info": partition_info}
    mapper = CachingVisitor(_fuse_ir_node, state=state)
    return mapper(ir)
