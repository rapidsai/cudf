# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Explain logical and physical plans."""

from __future__ import annotations

import functools
from collections.abc import MutableMapping
from itertools import groupby
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import (
    IR,
    GroupBy,
    Join,
    Scan,
    Sort,
)
from cudf_polars.dsl.translate import Translator
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    import polars as pl


def explain_query(
    q: pl.LazyFrame, engine: pl.GPUEngine, *, physical: bool = True
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
        lowered_ir, partition_info = lower_ir_graph(ir, config)
        return _repr_ir_tree(lowered_ir, partition_info)
    else:
        return _repr_ir_tree(ir)


def _repr_ir_tree(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo] | None = None,
    *,
    offset: str = "",
) -> str:
    header = _repr_ir(ir, partition_info, offset=offset)

    children_strs = [
        _repr_ir_tree(child, partition_info, offset=offset + "  ")
        for child in ir.children
    ]

    formatted = []
    for line, group in groupby(children_strs):
        count = sum(1 for _ in group)
        formatted.append(line)
        if count > 1:
            formatted.append(f"{offset}  (repeated {count} times)\n")

    return header + "".join(formatted)


@functools.singledispatch
def _repr_ir(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo] | None = None,
    *,
    offset: str = "",
) -> str:
    count = partition_info[ir].count if partition_info else None
    header = f"{offset}{type(ir).__name__.upper()}"

    schema = getattr(ir, "schema", None)
    if schema is not None:
        names = tuple(schema)
        if len(names) > 6:
            names = names[:3] + ("...",) + names[-2:]
        header += f" {names}"

    if count is not None:
        header += f" [{count}]"
    return header + "\n"


@_repr_ir.register
def _(
    ir: GroupBy,
    partition_info: MutableMapping[IR, PartitionInfo] | None = None,
    *,
    offset: str = "",
) -> str:
    keys = tuple(ne.name for ne in ir.keys)
    count = partition_info[ir].count if partition_info else None
    return (
        f"{offset}GROUPBY {keys}" + (f" [{count}]" if count is not None else "") + "\n"
    )


@_repr_ir.register
def _(
    ir: Join,
    partition_info: MutableMapping[IR, PartitionInfo] | None = None,
    *,
    offset: str = "",
) -> str:
    left_on = tuple(ne.name for ne in ir.left_on)
    right_on = tuple(ne.name for ne in ir.right_on)
    count = partition_info[ir].count if partition_info else None
    return (
        f"{offset}JOIN {ir.options[0]} {left_on} {right_on}"
        + (f" [{count}]" if count is not None else "")
        + "\n"
    )


@_repr_ir.register
def _(
    ir: Sort,
    partition_info: MutableMapping[IR, PartitionInfo] | None = None,
    *,
    offset: str = "",
) -> str:
    by = tuple(ne.name for ne in ir.by)
    count = partition_info[ir].count if partition_info else None
    return f"{offset}SORT {by}" + (f" [{count}]" if count is not None else "") + "\n"


@_repr_ir.register
def _(
    ir: Scan,
    partition_info: MutableMapping[IR, PartitionInfo] | None = None,
    *,
    offset: str = "",
) -> str:
    count = partition_info[ir].count if partition_info else None
    schema = tuple(ir.schema)
    first_path = ir.paths[0]
    suffix = " ..." if len(ir.paths) > 1 else ""
    header = f"{offset}SCAN {ir.typ.upper()} {schema} {first_path}{suffix}"
    if count is not None:
        header += f" [{count} partition{'s' if count > 1 else ''}]"
    return header + "\n"
