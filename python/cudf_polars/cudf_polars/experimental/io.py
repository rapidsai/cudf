# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel IO Logic."""

from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

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

    _STATS_CACHE: ClassVar[dict[int, dict[str, float]]] = {}

    def _sample_pq_statistics(self) -> dict[str, float]:
        import numpy as np
        import pyarrow.dataset as pa_ds

        n_sample = 3
        paths = self.paths[:n_sample]
        key = hash(tuple(paths))
        try:
            return self._STATS_CACHE[key]
        except KeyError:
            # Use average total_uncompressed_size of three files
            column_sizes = {}
            ds = pa_ds.dataset(paths, format="parquet")
            for i, frag in enumerate(ds.get_fragments()):
                md = frag.metadata
                for rg in range(md.num_row_groups):
                    row_group = md.row_group(rg)
                    for col in range(row_group.num_columns):
                        column = row_group.column(col)
                        name = column.path_in_schema
                        if name not in column_sizes:
                            column_sizes[name] = np.zeros(n_sample, dtype="int64")
                        column_sizes[name][i] += column.total_uncompressed_size

            self._STATS_CACHE[key] = {
                name: np.mean(sizes) for name, sizes in column_sizes.items()
            }
            return self._STATS_CACHE[key]

    @cached_property
    def _stride(self) -> int:
        if self.typ == "parquet":
            file_size: float = 0
            # TODO: Choose blocksize wisely, and make it configurable
            blocksize = 2 * 1024**3
            stats = self._sample_pq_statistics()
            for name in self.with_columns or []:
                file_size += stats[name]
            if file_size > 0:
                return max(int(blocksize / file_size), 1)
        # TODO: Use file sizes for csv/json
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
