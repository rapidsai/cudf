# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Single-partition LogicalPlan nodes."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import (
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
from cudf_polars.experimental.partitioned import PartitionInfo

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


def _single_partition_node(node: IR, rec) -> SPartitionwise:
    children = [rec(child) for child in node.children]

    if isinstance(node, PythonScan):
        return SParPythonScan(*node._ctor_arguments(children))
    elif isinstance(node, Scan):
        return SParScan(*node._ctor_arguments(children))
    elif isinstance(node, Cache):
        return SParCache(*node._ctor_arguments(children))
    elif isinstance(node, DataFrameScan):
        return SParDataFrameScan(*node._ctor_arguments(children))
    elif isinstance(node, Select):
        return SParSelect(*node._ctor_arguments(children))
    elif isinstance(node, Reduce):
        return SParReduce(*node._ctor_arguments(children))
    elif isinstance(node, GroupBy):
        return SParGroupBy(*node._ctor_arguments(children))
    elif isinstance(node, Join):
        return SParJoin(*node._ctor_arguments(children))
    elif isinstance(node, HStack):
        return SParHStack(*node._ctor_arguments(children))
    elif isinstance(node, Distinct):
        return SParDistinct(*node._ctor_arguments(children))
    elif isinstance(node, Sort):
        return SParSort(*node._ctor_arguments(children))
    elif isinstance(node, Slice):
        return SParSlice(*node._ctor_arguments(children))
    elif isinstance(node, Filter):
        return SParFilter(*node._ctor_arguments(children))
    elif isinstance(node, Projection):
        return SParProjection(*node._ctor_arguments(children))
    elif isinstance(node, MapFunction):
        return SParMapFunction(*node._ctor_arguments(children))
    elif isinstance(node, Union):
        return SParUnion(*node._ctor_arguments(children))
    elif isinstance(node, HConcat):
        return SParHConcat(*node._ctor_arguments(children))
    else:
        raise NotImplementedError(f"Cannot convert {type(node)} to PartitionedIR.")


lower_ir_graph = CachingVisitor(_single_partition_node)
