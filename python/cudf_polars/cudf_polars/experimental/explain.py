# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Explain logical and physical plans."""

from __future__ import annotations

import functools
from itertools import groupby
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import (
    GroupBy,
    Join,
    Scan,
    Sort,
)
from cudf_polars.dsl.translate import Translator
from cudf_polars.experimental.base import ColumnStat
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.statistics import (
    collect_statistics,
)
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    import polars as pl

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector


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
