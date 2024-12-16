# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle Logic."""

from __future__ import annotations

import json
import operator
from typing import TYPE_CHECKING, Any

import pyarrow as pa

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR
from cudf_polars.experimental.base import _concat, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks

if TYPE_CHECKING:
    from collections.abc import Hashable, MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.experimental.parallel import PartitionInfo
    from cudf_polars.typing import Schema


class Shuffle(IR):
    """Suffle multi-partition data."""

    __slots__ = ("keys", "options")
    _non_child = ("schema", "keys", "options")
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
        )

    @classmethod
    def do_evaluate(cls, df: DataFrame):  # pragma: no cover
        """Evaluate and return a dataframe."""
        # Single-partition logic is a no-op
        return df


def _split_on_columns(
    df: DataFrame,
    on: tuple[NamedExpr, ...],
    count: int,
) -> dict[int, DataFrame]:
    # Use murmurhash % count to choose the
    # destination partition id for each row.
    partition_map = plc.binaryop.binary_operation(
        plc.hashing.murmurhash3_x86_32(
            DataFrame([expr.evaluate(df) for expr in on]).table
        ),
        plc.interop.from_arrow(pa.scalar(count, type="uint32")),
        plc.binaryop.BinaryOperator.PYMOD,
        plc.types.DataType(plc.types.TypeId.UINT32),
    )

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
    on: tuple[NamedExpr, ...],
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
                _split_on_columns,
                (name_in, part_in),
                on,
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


@generate_ir_tasks.register(Shuffle)
def _(
    ir: Shuffle, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    child_ir = ir.children[0]
    return _simple_shuffle_graph(
        get_key_name(ir),
        get_key_name(child_ir),
        ir.keys,
        partition_info[child_ir].count,
        partition_info[ir].count,
    )
