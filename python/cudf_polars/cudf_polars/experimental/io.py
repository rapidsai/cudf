# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel IO Logic."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import Scan
from cudf_polars.experimental.parallel import (
    PartitionInfo,
    _ir_parts_info,
    generate_ir_tasks,
    get_key_name,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from polars import GPUEngine

    from cudf_polars.dsl.ir import IR


class ParFileScan(Scan):
    """Parallel scan over files."""

    @property
    def _stride(self) -> int:
        # TODO: Use file statistics
        return 1


def lower_scan_node(ir: Scan, rec) -> IR:
    """Rewrite a Scan node with proper partitioning."""
    if (
        len(ir.paths) > 1
        and ir.typ in ("csv", "parquet", "ndjson")
        and ir.n_rows == -1
        and ir.skip_rows == 0
    ):
        # TODO: mypy complains: ParFileScan(*ir._ctor_arguments([]))
        return ParFileScan(
            ir.schema,
            ir.typ,
            ir.reader_options,
            ir.cloud_options,
            ir.paths,
            ir.with_columns,
            ir.skip_rows,
            ir.n_rows,
            ir.row_index,
            ir.predicate,
        )
    return ir


@_ir_parts_info.register(ParFileScan)
def _(ir: ParFileScan) -> PartitionInfo:
    count = math.ceil(len(ir.paths) / ir._stride)
    return PartitionInfo(count=count)


@generate_ir_tasks.register(ParFileScan)
def _(ir: ParFileScan, config: GPUEngine) -> MutableMapping[Any, Any]:
    key_name = get_key_name(ir)
    stride = ir._stride
    paths = list(ir.paths)
    return {
        (key_name, i): (
            ir.do_evaluate,
            config,
            ir.schema,
            ir.typ,
            ir.reader_options,
            paths[j : j + stride],
            ir.with_columns,
            ir.skip_rows,
            ir.n_rows,
            ir.row_index,
            ir.predicate,
        )
        for i, j in enumerate(range(0, len(paths), stride))
    }
