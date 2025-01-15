# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle Logic."""

from __future__ import annotations

import json
import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

import pyarrow as pa

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR
from cudf_polars.experimental.base import _concat, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node

if TYPE_CHECKING:
    from collections.abc import Hashable, MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.experimental.parallel import PartitionInfo
    from cudf_polars.typing import Schema


class Shuffle(IR):
    """
    Suffle multi-partition data.

    Notes
    -----
    A Shuffle node may have either one or two children. In both
    cases, the first child corresponds to the DataFrame we are
    shuffling. The optional second child corresponds to a distinct
    DataFrame to extract the shuffle keys from. For example, it
    may be useful to reference a distinct DataFrame in the case
    of sorting.

    The type of argument `keys` controls whether or not hash
    partitioning will be applied. If `keys` is a tuple, we
    assume that the corresponding columns must be hashed. If
    `keys` is a `NamedExpr`, we assume that the corresponding
    column already contains a direct partition mapping.
    """

    __slots__ = ("keys", "options")
    _non_child = ("schema", "keys", "options")
    keys: tuple[NamedExpr, ...] | NamedExpr
    """Keys to shuffle on."""
    options: dict[str, Any]
    """Shuffling options."""

    def __init__(
        self,
        schema: Schema,
        keys: tuple[NamedExpr, ...] | NamedExpr,
        options: dict[str, Any],
        *children: IR,
    ):
        self.schema = schema
        self.keys = keys
        self.options = options
        self._non_child_args = ()
        self.children = children
        if len(children) > 2:  # pragma: no cover
            raise ValueError(f"Expected a maximum of two children, got {children}")

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


def _partition_dataframe(
    df: DataFrame,
    index: DataFrame | None,
    keys: tuple[NamedExpr, ...] | NamedExpr,
    count: int,
) -> dict[int, DataFrame]:
    """
    Partition an input DataFrame for shuffling.

    Parameters
    ----------
    df
        DataFrame to partition.
    index
        Optional DataFrame from which to extract partitioning
        keys. If None, keys will be extracted from `df`.
    keys
        Shuffle key(s) to extract from index or df.
    count
        Total number of output partitions.

    Returns
    -------
    A dictionary mapping between int partition indices and
    DataFrame fragments.
    """
    # Extract output-partition mapping
    if isinstance(keys, tuple):
        # Hash the specified keys to calculate the output
        # partition for each row (partition_map)
        partition_map = plc.binaryop.binary_operation(
            plc.hashing.murmurhash3_x86_32(
                DataFrame([expr.evaluate(index or df) for expr in keys]).table
            ),
            plc.interop.from_arrow(pa.scalar(count, type="uint32")),
            plc.binaryop.BinaryOperator.PYMOD,
            plc.types.DataType(plc.types.TypeId.UINT32),
        )
    else:  # pragma: no cover
        # Specified key column already contains the
        # output-partition index in each row
        partition_map = keys.evaluate(index or df).obj

    # Apply partitioning
    t, offsets = plc.partitioning.partition(
        df.table,
        partition_map,
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
    name_index: str | None,
    keys: tuple[NamedExpr, ...] | NamedExpr,
    count_in: int,
    count_out: int,
) -> MutableMapping[Any, Any]:
    """Make a simple all-to-all shuffle graph."""
    split_name = f"split-{name_out}"
    inter_name = f"inter-{name_out}"

    graph: MutableMapping[Any, Any] = {}
    for part_out in range(count_out):
        _concat_list = []
        for part_in in range(count_in):
            graph[(split_name, part_in)] = (
                _partition_dataframe,
                (name_in, part_in),
                None if name_index is None else (name_index, part_in),
                keys,
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


@lower_ir_node.register(Shuffle)
def _(
    ir: Shuffle, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Simple lower_ir_node handling for the default hash-based shuffle.
    # More-complex logic (e.g. joining and sorting) should
    # be handled separately.
    from cudf_polars.experimental.parallel import PartitionInfo

    # Check ir.keys
    if not isinstance(ir.keys, tuple):  # pragma: no cover
        raise NotImplementedError(
            f"Default hash Shuffle does not support NamedExpr keys argument. Got {ir.keys}"
        )

    # Extract child partitioning
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)
    pi = partition_info[children[0]]

    # Check child count
    if len(children) > 1:  # pragma: no cover
        raise NotImplementedError(
            f"Default hash Shuffle does not support multiple children. Got {children}"
        )

    # Check if we are already shuffled or update partition_info
    if ir.keys == pi.partitioned_on:
        # Already shuffled!
        new_node = children[0]
    else:
        new_node = ir.reconstruct(children)
        partition_info[new_node] = PartitionInfo(
            # Default shuffle preserves partition count
            count=pi.count,
            # Add partitioned_on info
            partitioned_on=ir.keys,
        )

    return new_node, partition_info


@generate_ir_tasks.register(Shuffle)
def _(
    ir: Shuffle, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    # Use a simple all-to-all shuffle graph.

    # TODO: Optionally use rapidsmp.
    if len(ir.children) > 1:  # pragma: no cover
        child_ir, index_ir = ir.children
        index_name = get_key_name(index_ir)
    else:
        child_ir = ir.children[0]
        index_name = None

    return _simple_shuffle_graph(
        get_key_name(ir),
        get_key_name(child_ir),
        index_name,
        ir.keys,
        partition_info[child_ir].count,
        partition_info[ir].count,
    )
