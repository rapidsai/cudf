# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle Logic."""

from __future__ import annotations

import json
import operator
from typing import TYPE_CHECKING, Any

import pyarrow as pa

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.dsl.ir import IR
from cudf_polars.experimental.base import _concat, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node

if TYPE_CHECKING:
    from collections.abc import Hashable, MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.experimental.parallel import PartitionInfo
    from cudf_polars.typing import Schema


class HashIndex(IR):
    """Construct a hash-based index for shuffling."""

    __slots__ = ("count", "keys")
    _non_child = ("schema", "keys", "count")
    keys: tuple[NamedExpr, ...]
    """Columns to hash partition on."""
    count: int
    """Number of output partitions."""

    def __init__(
        self,
        schema: Schema,
        keys: tuple[NamedExpr, ...],
        count: int,
        df: IR,
    ):
        self.schema = schema
        self.keys = keys
        self.count = count
        self._non_child_args = (keys, next(iter(schema)), count)
        self.children = (df,)

    @classmethod
    def do_evaluate(
        cls,
        keys: tuple[NamedExpr, ...],
        name: str,
        count: int,
        df: DataFrame,
    ):
        """Evaluate and return a dataframe."""
        partition_map = Column(
            plc.binaryop.binary_operation(
                plc.hashing.murmurhash3_x86_32(
                    DataFrame([expr.evaluate(df) for expr in keys]).table
                ),
                plc.interop.from_arrow(pa.scalar(count, type="uint32")),
                plc.binaryop.BinaryOperator.PYMOD,
                plc.types.DataType(plc.types.TypeId.UINT32),
            ),
            name=name,
        )
        return DataFrame([partition_map])


class ShuffleByIndex(IR):
    """Suffle multi-partition data by a partition index."""

    __slots__ = ("count", "options")
    _non_child = ("schema", "options", "count")
    options: dict[str, Any]
    """Shuffling options."""
    count: int
    """Number of output partitions."""

    def __init__(
        self,
        schema: Schema,
        options: dict[str, Any],
        count: int,
        df: IR,
        partition_index: IR,
    ):
        self.schema = schema
        self.options = options
        self.count = count
        self._non_child_args = (count,)
        self.children = (df, partition_index)

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        return (
            type(self),
            tuple(self.schema.items()),
            json.dumps(self.options),
            self.count,
            self.children,
        )

    @classmethod
    def do_evaluate(
        cls, count: int, df: DataFrame, partition_index: DataFrame
    ):  # pragma: no cover
        """Evaluate and return a dataframe."""
        # Single-partition logic is a no-op
        return df


class ShuffleByHash(IR):
    """Suffle data by hash partitioning."""

    __slots__ = ("keys", "options")
    _non_child = ("schema", "options", "keys")
    keys: tuple[NamedExpr, ...]
    """Columns to shuffle on."""
    options: dict[str, Any]
    """Shuffling options."""

    def __init__(
        self,
        schema: Schema,
        keys: tuple[NamedExpr, ...],
        options: dict[str, Any],
        df: IR,
    ):
        self.schema = schema
        self.keys = keys
        self.options = options
        self._non_child_args = ()
        self.children = (df,)

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        return (
            type(self),
            tuple(self.schema.items()),
            self.keys,
            json.dumps(self.options),
            self.children,
        )

    @classmethod
    def do_evaluate(cls, df: DataFrame):  # pragma: no cover
        """Evaluate and return a dataframe."""
        # Single-partition logic is a no-op
        return df


def _split_by_index(
    df: DataFrame,
    index: DataFrame,
    count: int,
) -> dict[int, DataFrame]:
    # Apply partitioning
    assert len(index.column_map) == 1, "Partition index has too many columns."
    t, offsets = plc.partitioning.partition(
        df.table,
        next(iter(index.column_map.values())).obj,
        count,
    )

    # Split and return the partitioned result
    return {
        i: DataFrame.from_table(
            split,
            df.column_names,
        )
        for i, split in enumerate(plc.copying.split(t, offsets[1:-1]))
    }


def _simple_shuffle_graph(
    name_out: str,
    name_in: str,
    name_index: str,
    count_in: int,
    count_out: int,
) -> MutableMapping[Any, Any]:
    """Make a simple all-to-all shuffle graph."""
    # Simple all-to-all shuffle (for now)
    split_name = f"split-{name_out}"
    inter_name = f"inter-{name_out}"

    graph: MutableMapping[Any, Any] = {}
    for part_out in range(count_out):
        _concat_list = []
        for part_in in range(count_in):
            graph[(split_name, part_in)] = (
                _split_by_index,
                (name_in, part_in),
                (name_index, part_in),
                count_out,
            )
            _concat_list.append((inter_name, part_out, part_in))
            graph[_concat_list[-1]] = (
                operator.getitem,
                (split_name, part_in),
                part_out,
            )
        graph[(name_out, part_out)] = (_concat, _concat_list)
    return graph


@lower_ir_node.register(ShuffleByHash)
def _(
    ir: ShuffleByHash, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    from cudf_polars.experimental.parallel import PartitionInfo

    # Extract child partitioning
    child, partition_info = rec(ir.children[0])
    pi = partition_info[child]

    # Add a HashIndex node
    partition_index = HashIndex(
        {"_shuffle_index": plc.types.DataType(plc.types.TypeId.UINT32)},
        ir.keys,
        pi.count,
        child,
    )
    partition_info[partition_index] = PartitionInfo(count=pi.count)

    # Shuffle by the HashIndex node
    new_node = ShuffleByIndex(
        child.schema,
        ir.options,
        pi.count,
        child,
        partition_index,
    )
    partition_info[new_node] = PartitionInfo(count=pi.count)
    return new_node, partition_info


@generate_ir_tasks.register(HashIndex)
def _(
    ir: HashIndex, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    # HashIndex is a partition-wise operation
    from cudf_polars.experimental.parallel import _generate_ir_tasks_pwise

    return _generate_ir_tasks_pwise(ir, partition_info)


@generate_ir_tasks.register(ShuffleByIndex)
def _(
    ir: ShuffleByIndex, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    # Use a simple all-to-all shuffle graph.
    # TODO: Optionally use rapidsmp.
    child_ir, index_ir = ir.children
    return _simple_shuffle_graph(
        get_key_name(ir),
        get_key_name(child_ir),
        get_key_name(index_ir),
        partition_info[child_ir].count,
        partition_info[ir].count,
    )
