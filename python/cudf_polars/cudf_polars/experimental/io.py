# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel IO Logic."""

from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.dsl.ir import DataFrameScan, Scan
from cudf_polars.experimental.parallel import (
    PartitionInfo,
    _default_lower_ir_node,
    generate_ir_tasks,
    get_key_name,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


##
## DataFrameScan
##


class ParDataFrameScan(DataFrameScan):
    """Parallel DataFrameScan."""

    @property
    def _max_n_rows(self) -> int:
        """Row-count threshold for splitting a DataFrame."""
        parallel_options = self.config_options.get("parallel_options", {})
        return parallel_options.get("num_rows_threshold", 1_000_000)

    @property
    def _count(self) -> int:
        """Partition count."""
        total_rows, _ = self.df.shape()
        return math.ceil(total_rows / self._max_n_rows)

    def _tasks(self) -> MutableMapping[Any, Any]:
        """Task graph."""
        total_rows, _ = self.df.shape()
        stride = math.ceil(total_rows / self._count)
        key_name = get_key_name(self)
        return {
            (key_name, i): (
                self.do_evaluate,
                self.schema,
                self.df.slice(offset, stride),
                self.projection,
                self.predicate,
            )
            for i, offset in enumerate(range(0, total_rows, stride))
        }


def lower_dataframescan_node(
    ir: DataFrameScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Rewrite a Scan node with proper partitioning."""
    new_node = ParDataFrameScan(
        ir.schema,
        ir.df,
        ir.projection,
        ir.predicate,
        ir.config_options,
    )
    return new_node, {new_node: PartitionInfo(count=new_node._count)}


@generate_ir_tasks.register(ParDataFrameScan)
def _(
    ir: ParDataFrameScan, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    return ir._tasks()


##
## Scan
##


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
            # TODO: Use plc.io.parquet_metadata.read_parquet_metadata
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
    def _plan(self) -> tuple[int, int]:
        split, stride = 1, 1
        if self.typ == "parquet":
            file_size: float = 0
            # TODO: Use system info to set default blocksize
            parallel_options = self.config_options.get("parallel_options", {})
            blocksize: int = parallel_options.get("parquet_blocksize", 2 * 1024**3)
            stats = self._sample_pq_statistics()
            columns: list = self.with_columns or list(stats.keys())
            for name in columns:
                file_size += float(stats[name])
            if file_size > 0:
                if file_size > blocksize:
                    # Split large files
                    split = math.ceil(file_size / blocksize)
                else:
                    # Aggregate small files
                    stride = max(int(blocksize / file_size), 1)

        # TODO: Use file sizes for csv/json?
        return (split, stride)


def lower_scan_node(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Rewrite a Scan node with proper partitioning."""
    if ir.typ in ("csv", "parquet", "ndjson") and ir.n_rows == -1 and ir.skip_rows == 0:
        # TODO: mypy complains: ParFileScan(*ir._ctor_arguments([]))
        new_node = ParFileScan(
            ir.schema,
            ir.typ,
            ir.reader_options,
            ir.cloud_options,
            ir.config_options,
            ir.paths,
            ir.with_columns,
            ir.skip_rows,
            ir.n_rows,
            ir.row_index,
            ir.predicate,
        )
        split, stride = new_node._plan
        if split > 1:
            count = len(new_node.paths) * split
        else:
            count = math.ceil(len(new_node.paths) / stride)
        return new_node, {new_node: PartitionInfo(count=count)}

    return _default_lower_ir_node(ir, rec)


def _split_read(
    do_evaluate,
    split_index,
    total_splits,
    schema,
    typ,
    reader_options,
    config_options,
    paths,
    with_columns,
    skip_rows,
    n_rows,
    row_index,
    predicate,
):
    if typ != "parquet":
        raise NotImplementedError()

    rowgroup_metadata = plc.io.parquet_metadata.read_parquet_metadata(
        plc.io.SourceInfo(paths)
    ).rowgroup_metadata()
    total_row_groups = len(rowgroup_metadata)
    if total_splits > total_row_groups:
        # Don't bother aligning on row-groups
        total_rows = sum(rg["num_rows"] for rg in rowgroup_metadata)
        n_rows = int(total_rows / total_splits)
        skip_rows = n_rows * split_index
    else:
        # Align split with row-groups
        rg_stride = int(total_row_groups / total_splits)
        skip_rgs = rg_stride * split_index
        skip_rows = (
            sum(rg["num_rows"] for rg in rowgroup_metadata[:skip_rgs])
            if skip_rgs
            else 0
        )
        n_rows = sum(
            rg["num_rows"] for rg in rowgroup_metadata[skip_rgs : skip_rgs + rg_stride]
        )

    # Last split should always read to end of file
    if split_index == (total_splits - 1):
        n_rows = -1

    return do_evaluate(
        schema,
        typ,
        reader_options,
        config_options,
        paths,
        with_columns,
        skip_rows,
        n_rows,
        row_index,
        predicate,
    )


@generate_ir_tasks.register(ParFileScan)
def _(
    ir: ParFileScan, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    key_name = get_key_name(ir)
    split, stride = ir._plan
    paths = list(ir.paths)
    if split > 1:
        # Disable chunked reader
        config_options = ir.config_options.copy()
        config_options["parquet_options"] = config_options.get(
            "parquet_options", {}
        ).copy()
        config_options["parquet_options"]["chunked"] = False

        graph = {}
        count = 0
        for path in paths:
            for sindex in range(split):
                graph[(key_name, count)] = (
                    _split_read,
                    ir.do_evaluate,
                    sindex,
                    split,
                    ir.schema,
                    ir.typ,
                    ir.reader_options,
                    config_options,
                    [path],
                    ir.with_columns,
                    ir.skip_rows,
                    ir.n_rows,
                    ir.row_index,
                    ir.predicate,
                )
                count += 1
        return graph
    else:
        return {
            (key_name, i): (
                ir.do_evaluate,
                ir.schema,
                ir.typ,
                ir.reader_options,
                ir.config_options,
                paths[j : j + stride],
                ir.with_columns,
                ir.skip_rows,
                ir.n_rows,
                ir.row_index,
                ir.predicate,
            )
            for i, j in enumerate(range(0, len(paths), stride))
        }
