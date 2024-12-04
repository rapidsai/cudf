# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition IO Logic."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.dsl.ir import IR, DataFrameScan, Scan, Union
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node

if TYPE_CHECKING:
    from collections.abc import Hashable, MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema


@lower_ir_node.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    rows_per_partition = ir.config_options.get("executor_options", {}).get(
        "max_rows_per_partition", 1_000_000
    )

    nrows = max(ir.df.shape()[0], 1)
    count = math.ceil(nrows / rows_per_partition)

    if count > 1:
        length = math.ceil(nrows / count)
        slices = [
            DataFrameScan(
                ir.schema,
                ir.df.slice(offset, length),
                ir.projection,
                ir.predicate,
                ir.config_options,
            )
            for offset in range(0, nrows, length)
        ]
        new_node = Union(ir.schema, None, *slices)
        return new_node, {slice: PartitionInfo(count=1) for slice in slices} | {
            new_node: PartitionInfo(count=count)
        }

    return ir, {ir: PartitionInfo(count=1)}


class SplitScan(IR):
    """Input from a split file."""

    __slots__ = (
        "base_scan",
        "schema",
        "split_index",
        "total_splits",
    )
    _non_child = (
        "base_scan",
        "split_index",
        "total_splits",
    )
    base_scan: Scan
    """Scan operation this node is based on."""
    split_index: int
    """Index of the current split."""
    total_splits: int
    """Total number of splits."""

    def __init__(self, base_scan: Scan, split_index: int, total_splits: int):
        self.schema = base_scan.schema
        self.base_scan = base_scan
        self.split_index = split_index
        self.total_splits = total_splits
        self._non_child_args = (
            split_index,
            total_splits,
            *base_scan._non_child_args,
        )
        self.children = ()
        if base_scan.typ not in ("parquet",):  # pragma: no cover
            raise NotImplementedError(
                f"Unhandled Scan type for file splitting: {base_scan.typ}"
            )

    def get_hashable(self) -> Hashable:
        """Hashable representation of node."""
        return (
            hash(self.base_scan),
            self.split_index,
            self.total_splits,
        )

    @classmethod
    def do_evaluate(
        cls,
        split_index: int,
        total_splits: int,
        schema: Schema,
        typ: str,
        reader_options: dict[str, Any],
        config_options: dict[str, Any],
        paths: list[str],
        with_columns: list[str] | None,
        skip_rows: int,
        n_rows: int,
        row_index: tuple[str, int] | None,
        predicate: NamedExpr | None,
    ):
        """Evaluate and return a dataframe."""
        if typ not in ("parquet",):  # pragma: no cover
            raise NotImplementedError(f"Unhandled Scan type for file splitting: {typ}")

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
                rg["num_rows"]
                for rg in rowgroup_metadata[skip_rgs : skip_rgs + rg_stride]
            )

        # Last split should always read to end of file
        if split_index == (total_splits - 1):
            n_rows = -1

        return Scan.do_evaluate(
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


def _sample_pq_statistics(ir: Scan) -> dict[str, float]:
    import numpy as np
    import pyarrow.dataset as pa_ds

    # Use average total_uncompressed_size of three files
    # TODO: Use plc.io.parquet_metadata.read_parquet_metadata
    n_sample = 3
    column_sizes = {}
    ds = pa_ds.dataset(ir.paths[:n_sample], format="parquet")
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

    return {name: np.mean(sizes) for name, sizes in column_sizes.items()}


def _scan_partitioning(ir: Scan) -> tuple[int, int]:
    split, stride = 1, 1
    if ir.typ == "parquet":
        file_size: float = 0
        # TODO: Use system info to set default blocksize
        parallel_options = ir.config_options.get("executor_options", {})
        blocksize: int = parallel_options.get("parquet_blocksize", 1024**3)
        stats = _sample_pq_statistics(ir)
        columns: list = ir.with_columns or list(stats.keys())
        for name in columns:
            file_size += float(stats[name])
        if file_size > 0:
            if file_size > blocksize:
                # Split large files
                split = math.ceil(file_size / blocksize)
            else:
                # Aggregate small files
                stride = max(int(blocksize / file_size), 1)

    # TODO: Use file sizes for csv and json
    return (split, stride)


@lower_ir_node.register(Scan)
def _(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    partition_info: MutableMapping[IR, PartitionInfo]
    if ir.typ in ("csv", "parquet", "ndjson") and ir.n_rows == -1 and ir.skip_rows == 0:
        split, stride = _scan_partitioning(ir)
        paths = list(ir.paths)
        if split > 1:
            # Disable chunked reader when splitting files
            config_options = ir.config_options.copy()
            config_options["parquet_options"] = config_options.get(
                "parquet_options", {}
            ).copy()
            config_options["parquet_options"]["chunked"] = False

            slices: list[SplitScan] = []
            for path in paths:
                base_scan = Scan(
                    ir.schema,
                    ir.typ,
                    ir.reader_options,
                    ir.cloud_options,
                    config_options,
                    [path],
                    ir.with_columns,
                    ir.skip_rows,
                    ir.n_rows,
                    ir.row_index,
                    ir.predicate,
                )
                slices.extend(
                    SplitScan(base_scan, sindex, split) for sindex in range(split)
                )
            new_node = Union(ir.schema, None, *slices)
            partition_info = {slice: PartitionInfo(count=1) for slice in slices} | {
                new_node: PartitionInfo(count=len(slices))
            }
        else:
            groups: list[Scan] = [
                Scan(
                    ir.schema,
                    ir.typ,
                    ir.reader_options,
                    ir.cloud_options,
                    ir.config_options,
                    paths[i : i + stride],
                    ir.with_columns,
                    ir.skip_rows,
                    ir.n_rows,
                    ir.row_index,
                    ir.predicate,
                )
                for i in range(0, len(paths), stride)
            ]
            new_node = Union(ir.schema, None, *groups)
            partition_info = {group: PartitionInfo(count=1) for group in groups} | {
                new_node: PartitionInfo(count=len(groups))
            }
        return new_node, partition_info

    return ir, {ir: PartitionInfo(count=1)}  # pragma: no cover
