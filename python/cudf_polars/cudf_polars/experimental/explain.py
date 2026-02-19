# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Explain logical and physical plans."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from itertools import groupby
from typing import TYPE_CHECKING, Self, TypeAlias

import cudf_polars.dsl.expressions.binaryop
import cudf_polars.dsl.expressions.literal
from cudf_polars.dsl.ir import (
    Filter,
    GroupBy,
    HStack,
    Join,
    Scan,
    Select,
    Sort,
)
from cudf_polars.dsl.translate import Translator
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import ColumnStat
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.statistics import (
    collect_statistics,
)
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    import polars as pl

    from cudf_polars.dsl.expressions.base import Expr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector


Serializable: TypeAlias = (
    str
    | int
    | float
    | bool
    | Sequence["Serializable"]
    | Mapping[str, "Serializable"]
    | None
)


def explain_query(
    q: pl.LazyFrame,
    engine: pl.GPUEngine,
    *,
    physical: bool = True,
) -> str:
    """
    Return a formatted string representation of the IR plan.

    Parameters
    ----------
    q : pl.LazyFrame
        The LazyFrame to explain.
    engine : pl.GPUEngine
        The configured GPU engine to use.
    physical : bool, default True
        If True, show the physical (lowered) plan.
        If False, show the logical (pre-lowering) plan.

    Returns
    -------
    str
        A string representation of the IR plan.
    """
    config = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()

    if physical:
        if (
            config.executor.name == "streaming"
            and config.executor.runtime == "rapidsmpf"
        ):  # pragma: no cover; rapidsmpf runtime not tested in CI yet
            from cudf_polars.experimental.rapidsmpf.core import (
                lower_ir_graph as rapidsmpf_lower_ir_graph,
            )

            lowered_ir, partition_info, _ = rapidsmpf_lower_ir_graph(ir, config)
        else:
            lowered_ir, partition_info, _ = lower_ir_graph(ir, config)
        return _repr_ir_tree(lowered_ir, partition_info)
    else:
        if config.executor.name == "streaming":
            # Include row-count statistics for the logical plan
            return _repr_ir_tree(ir, stats=collect_statistics(ir, config))
        else:
            return _repr_ir_tree(ir)


def serialize_query(
    q: pl.LazyFrame,
    engine: pl.GPUEngine,
    *,
    physical: bool = True,
) -> SerializablePlan:
    """
    Return a structured, serializable representation of the IR plan.

    Parameters
    ----------
    q : pl.LazyFrame
        The LazyFrame to serialize.
    engine : pl.GPUEngine
        The configured GPU engine to use.
    physical : bool, default True
        If True, serialize the physical (lowered) plan with partition info.
        If False, serialize the logical (pre-lowering) plan.

    Returns
    -------
    plan
        A structured representation of the query plan that can be
        serialized to JSON.

    Examples
    --------
    >>> import polars as pl
    >>> import json
    >>> import dataclasses
    >>> q = pl.LazyFrame({"a": [1, 2, 3]}).select(pl.col("a") * 2)
    >>> engine = pl.GPUEngine(executor="streaming")
    >>> plan = serialize_query(q, engine, physical=False)
    >>> print(json.dumps(dataclasses.asdict(plan), indent=2))
    {
      "roots": [
        "1739020873"
      ],
      "nodes": {
        "1739020873": {
          "id": "1739020873",
          "children": [
            "2653195019"
          ],
          "schema": {
            "a": "INT64"
          },
          "properties": {
            "columns": [
              "a"
            ]
          },
          "type": "Select"
        },
        "2653195019": {
          "id": "2653195019",
          "children": [],
          "schema": {
            "a": "INT64"
          },
          "properties": {},
          "type": "DataFrameScan"
        }
      },
      "partition_info": null
    }
    """
    return SerializablePlan.from_query(q, engine, lowered=physical)


def _fmt_row_count(value: int | None) -> str:
    """Format a row count as a readable string."""
    if value is None:
        return ""
    elif value < 1_000:
        return f"{value}"
    elif value < 1_000_000:
        return f"{round(value / 1_000, 2):g} K"
    elif value < 1_000_000_000:
        return f"{round(value / 1_000_000, 2):g} M"
    else:
        return f"{round(value / 1_000_000_000, 2):g} B"


def _repr_ir_tree(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo] | None = None,
    *,
    offset: str = "",
    stats: StatsCollector | None = None,
) -> str:
    header = _repr_ir(ir, offset=offset)
    count = partition_info[ir].count if partition_info else None
    if stats is not None:
        # Include row-count estimate (if available)
        row_count_estimate = _fmt_row_count(
            stats.row_count.get(ir, ColumnStat[int](None)).value
        )
        row_count = f"~{row_count_estimate}" if row_count_estimate else "unknown"
        header = header.rstrip("\n") + f" {row_count=}\n"
    if count is not None:
        header = header.rstrip("\n") + f" [{count}]\n"

    children_strs = [
        _repr_ir_tree(child, partition_info, offset=offset + "  ", stats=stats)
        for child in ir.children
    ]

    return header + "".join(
        f"{line}{offset}  (repeated {count} times)\n"
        if (count := sum(1 for _ in group)) > 1
        else line
        for line, group in groupby(children_strs)
    )


def _repr_schema(schema: tuple | None) -> str:
    if schema is None:
        return ""  # pragma: no cover; no test yet
    names = tuple(schema)
    if len(names) > 6:
        names = names[:3] + ("...",) + names[-2:]
    return f" {names}"


def _repr_header(offset: str, label: str, schema: tuple | dict | None) -> str:
    return f"{offset}{label}{_repr_schema(tuple(schema) if schema is not None else None)}\n"


@functools.singledispatch
def _repr_ir(ir: IR, *, offset: str = "") -> str:
    return _repr_header(offset, type(ir).__name__.upper(), ir.schema)


@_repr_ir.register
def _(ir: GroupBy, *, offset: str = "") -> str:
    keys = tuple(ne.name for ne in ir.keys)
    return _repr_header(offset, f"GROUPBY {keys}", ir.schema)


@_repr_ir.register
def _(ir: Join, *, offset: str = "") -> str:
    left_on = tuple(ne.name for ne in ir.left_on)
    right_on = tuple(ne.name for ne in ir.right_on)
    return _repr_header(offset, f"JOIN {ir.options[0]} {left_on} {right_on}", ir.schema)


@_repr_ir.register
def _(ir: Sort, *, offset: str = "") -> str:
    by = tuple(ne.name for ne in ir.by)
    return _repr_header(offset, f"SORT {by}", ir.schema)


@_repr_ir.register
def _(ir: Scan, *, offset: str = "") -> str:
    label = f"SCAN {ir.typ.upper()}"
    return _repr_header(offset, label, ir.schema)


# --------------------------------------------------------------------------
# Property serialization for structured query plan export
# --------------------------------------------------------------------------


@functools.singledispatch
def _serialize_properties(ir: IR) -> dict[str, Serializable]:
    """Extract serializable properties from an IR node."""
    return {}


@_serialize_properties.register
def _(ir: Scan) -> dict[str, Serializable]:
    # for polars<1.31, paths is a list[Path]
    # for polars>=1.31, paths is a list[str]
    return {
        "typ": ir.typ,
        "paths": [str(path) for path in ir.paths],
    }


@_serialize_properties.register
def _(ir: Join) -> dict[str, Serializable]:
    return {
        "how": ir.options[0],
        "left_on": [ne.name for ne in ir.left_on],
        "right_on": [ne.name for ne in ir.right_on],
    }


@_serialize_properties.register
def _(ir: GroupBy) -> dict[str, Serializable]:
    return {
        "keys": [ne.name for ne in ir.keys],
    }


@_serialize_properties.register
def _(ir: Sort) -> dict[str, Serializable]:
    return {
        "by": [ne.name for ne in ir.by],
        "order": [o.name for o in ir.order],
    }


def _serialize_expr(expr: Expr) -> dict[str, Serializable]:
    match expr:
        case cudf_polars.dsl.expressions.base.Col(name=name):
            return {"type": "Col", "name": name}
        case cudf_polars.dsl.expressions.literal.Literal(value=value):
            return {"type": "Literal", "value": value}
        case cudf_polars.dsl.expressions.binaryop.BinOp():
            return {
                "op": expr.op.name,
                "left": _serialize_expr(expr.children[0]),
                "right": _serialize_expr(expr.children[1]),
            }
        case _:  # pragma: no cover
            return {"type": type(expr).__name__}


@_serialize_properties.register
def _(ir: Filter) -> dict[str, Serializable]:
    value = ir.mask.value
    properties = _serialize_expr(value)
    properties["predicate"] = ir.mask.name

    return properties


@_serialize_properties.register
def _(ir: Select) -> dict[str, Serializable]:
    return {
        "columns": [ne.name for ne in ir.exprs],
    }


@_serialize_properties.register
def _(ir: HStack) -> dict[str, Serializable]:
    return {
        "columns": [ne.name for ne in ir.columns],
    }


@dataclasses.dataclass
class SerializableIRNode:
    """
    A node in the plan.

    This node is *serializable* and cannot be executed like a
    cudf_polars.dsl.ir.IR node.
    """

    id: str
    children: list[str]
    schema: dict[str, Serializable]
    properties: dict[str, Serializable]
    type: str

    @classmethod
    def from_ir(cls, ir: IR) -> Self:
        """Build a Node from an IR Node."""
        return cls(
            id=str(ir.get_stable_id()),
            children=[str(child.get_stable_id()) for child in ir.children],
            schema={k: v.id().name for k, v in ir.schema.items()},
            properties=_serialize_properties(ir),
            type=type(ir).__name__,
        )


@dataclasses.dataclass
class SerializablePartitionInfo:
    """Serializable information about a partition."""

    count: int
    partitioned_on: tuple[Serializable, ...]


@dataclasses.dataclass
class SerializablePlan:
    """
    A serializable representation of a query plan.

    Parameters
    ----------
    roots
        The IDs of the root nodes of the plan.
    nodes
        A mapping from node ID to node details.
    partition_info
        Information about the partitions of the plan.

    Notes
    -----
    All integers node IDs are stored as strings to make round-tripping
    to JSON easier. Node IDs will appear in

    - ``roots``
    - the keys of ``nodes``
    - the ``children`` of each node in ``nodes``
    - the keys in ``partition_info``

    You can safely rely on every key being present in ``nodes``.

    See Also
    --------
    serialize_query
        A function that builds a serializable plan from a LazyFrame query.
    """

    roots: list[str]
    nodes: dict[str, SerializableIRNode]
    partition_info: dict[str, SerializablePartitionInfo] | None = None

    @classmethod
    def from_ir(
        cls, ir: IR, *, config_options: ConfigOptions, lowered: bool = False
    ) -> Self:
        """
        Construct a serializable plan from an IR node.

        Parameters
        ----------
        ir
            The IR node to construct the serializable plan from.
        config_options
            The configuration options.
        lowered
            If True, lower the IR to the physical plan and include partition info.

        Returns
        -------
        plan
            A serializable representation of the query plan.
        """
        partition_info_dict: dict[str, SerializablePartitionInfo] | None = None
        if lowered:
            if (
                config_options.executor.name == "streaming"
                and config_options.executor.runtime == "rapidsmpf"
            ):  # pragma: no cover; rapidsmpf runtime not tested in CI yet
                from cudf_polars.experimental.rapidsmpf.core import (
                    lower_ir_graph as rapidsmpf_lower_ir_graph,
                )

                ir, partition_info_d, _ = rapidsmpf_lower_ir_graph(ir, config_options)
            else:
                ir, partition_info_d, _ = lower_ir_graph(ir, config_options)
            partition_info_dict = {}

        nodes: dict[str, SerializableIRNode] = {}
        for ir_node in traversal([ir]):
            stable_id = str(ir_node.get_stable_id())
            nodes[stable_id] = SerializableIRNode.from_ir(ir_node)
            if partition_info_dict is not None:
                partition_info_dict[stable_id] = SerializablePartitionInfo(
                    count=partition_info_d[ir_node].count,
                    partitioned_on=tuple(
                        expr.name for expr in partition_info_d[ir_node].partitioned_on
                    ),
                )

        return cls(
            roots=[str(ir.get_stable_id())],
            nodes=nodes,
            partition_info=partition_info_dict,
        )

    @classmethod
    def from_query(
        cls,
        q: pl.LazyFrame,
        engine: pl.GPUEngine,
        *,
        lowered: bool = False,
    ) -> Self:
        """
        Build a serializable plan from a LazyFrame query.

        Parameters
        ----------
        q
            The LazyFrame to serialize.
        engine
            The GPU engine to use. If None, uses default streaming executor.
        lowered
            If True, lower the IR to the physical plan and include partition info.

        Returns
        -------
        plan
            A serializable representation of the query plan.
        """
        config_options = ConfigOptions.from_polars_engine(engine)
        ir = Translator(q._ldf.visit(), engine).translate_ir()
        return cls.from_ir(ir, config_options=config_options, lowered=lowered)
