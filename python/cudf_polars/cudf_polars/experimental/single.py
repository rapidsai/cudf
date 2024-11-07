# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Single-partition LogicalPlan nodes."""

from __future__ import annotations

from functools import cached_property, singledispatch
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import (
    IR,
    Cache,
    DataFrameScan,
    Distinct,
    Filter,
    GroupBy,
    HConcat,
    HStack,
    Join,
    MapFunction,
    Projection,
    PythonScan,
    Reduce,
    Scan,
    Select,
    Slice,
    Sort,
    Union,
)
from cudf_polars.dsl.traversal import CachingVisitor
from cudf_polars.experimental.parallel import PartitionInfo

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR


class SPartitionwise:
    """Single partition-wise PartitionedIR."""

    @cached_property
    def _key(self):
        return f"{type(self).__name__.lower()}-{hash(self)}"

    def _tasks(self):
        return {
            (self._key, 0): (
                self.do_evaluate,
                *self._non_child_args,
                *((child._key, 0) for child in self.children),
            )
        }

    @cached_property
    def _parts(self) -> PartitionInfo:
        return PartitionInfo(npartitions=1)


class SParPythonScan(PythonScan, SPartitionwise):
    """Single-partition demo class."""


class SParScan(Scan, SPartitionwise):
    """Single-partition demo class."""


class SParCache(Cache, SPartitionwise):
    """Single-partition demo class."""


class SParDataFrameScan(DataFrameScan, SPartitionwise):
    """Single-partition demo class."""


class SParSelect(Select, SPartitionwise):
    """Single-partition demo class."""


class SParReduce(Reduce, SPartitionwise):
    """Single-partition demo class."""


class SParGroupBy(GroupBy, SPartitionwise):
    """Single-partition demo class."""


class SParJoin(Join, SPartitionwise):
    """Single-partition demo class."""


class SParHStack(HStack, SPartitionwise):
    """Single-partition demo class."""


class SParDistinct(Distinct, SPartitionwise):
    """Single-partition demo class."""


class SParSort(Sort, SPartitionwise):
    """Single-partition demo class."""


class SParSlice(Slice, SPartitionwise):
    """Single-partition demo class."""


class SParFilter(Filter, SPartitionwise):
    """Single-partition demo class."""


class SParProjection(Projection, SPartitionwise):
    """Single-partition demo class."""


class SParMapFunction(MapFunction, SPartitionwise):
    """Single-partition demo class."""


class SParUnion(Union, SPartitionwise):
    """Single-partition demo class."""


class SParHConcat(HConcat, SPartitionwise):
    """Single-partition demo class."""


@singledispatch
def _single_partition_node(node: IR, rec) -> SPartitionwise:
    raise NotImplementedError(f"Cannot convert {type(node)} to PartitionedIR.")


@_single_partition_node.register
def _(node: PythonScan, rec) -> SParPythonScan:
    return SParPythonScan(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Scan, rec) -> SParScan:
    return SParScan(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: DataFrameScan, rec) -> SParDataFrameScan:
    return SParDataFrameScan(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Cache, rec) -> SParCache:
    return SParCache(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Reduce, rec) -> SParReduce:
    return SParReduce(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Select, rec) -> SParSelect:
    return SParSelect(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: GroupBy, rec) -> SParGroupBy:
    return SParGroupBy(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Join, rec) -> SParJoin:
    return SParJoin(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: HStack, rec) -> SParHStack:
    return SParHStack(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Distinct, rec) -> SParDistinct:
    return SParDistinct(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Sort, rec) -> SParSort:
    return SParSort(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Slice, rec) -> SParSlice:
    return SParSlice(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Filter, rec) -> SParFilter:
    return SParFilter(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Projection, rec) -> SParProjection:
    return SParProjection(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: MapFunction, rec) -> SParMapFunction:
    return SParMapFunction(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: Union, rec) -> SParUnion:
    return SParUnion(*node._ctor_arguments(map(rec, node.children)))


@_single_partition_node.register
def _(node: HConcat, rec) -> SParHConcat:
    return SParHConcat(*node._ctor_arguments(map(rec, node.children)))


lower_ir_graph = CachingVisitor(_single_partition_node)
