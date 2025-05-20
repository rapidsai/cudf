# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition IO Logic."""

from __future__ import annotations

import dataclasses
import enum
import itertools
import math
from enum import IntEnum
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

import pylibcudf as plc

from cudf_polars.dsl.ir import IR, DataFrameScan, Scan, Union
from cudf_polars.experimental.base import ColumnStats, PartitionInfo, TableStats
from cudf_polars.experimental.dispatch import lower_ir_node

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    import numpy.typing as npt

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions

    T = TypeVar("T", bound=npt.NBitBase)


# Cache TableStats for each tuple of path names
_TABLESTATS_CACHE: MutableMapping[tuple[str, ...], TableStats] = {}


@lower_ir_node.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    assert ir.config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_tasks'"
    )

    rows_per_partition = ir.config_options.executor.max_rows_per_partition

    nrows = max(ir.df.shape()[0], 1)
    count = math.ceil(nrows / rows_per_partition)

    table_stats = TableStats(column_stats={}, num_rows=nrows)
    if count > 1:
        length = math.ceil(nrows / count)
        slices = [
            DataFrameScan(
                ir.schema,
                ir.df.slice(offset, length),
                ir.projection,
                ir.config_options,
            )
            for offset in range(0, nrows, length)
        ]
        new_node = Union(ir.schema, None, *slices)
        return new_node, {slice: PartitionInfo(count=1) for slice in slices} | {
            new_node: PartitionInfo(count=count, table_stats=table_stats)
        }

    return ir, {ir: PartitionInfo(count=1, table_stats=table_stats)}


class ScanPartitionFlavor(IntEnum):
    """Flavor of Scan partitioning."""

    SINGLE_FILE = enum.auto()  # 1:1 mapping between files and partitions
    SPLIT_FILES = enum.auto()  # Split each file into >1 partition
    FUSED_FILES = enum.auto()  # Fuse multiple files into each partition


class ScanPartitionPlan:
    """
    Scan partitioning plan.

    Notes
    -----
    The meaning of `factor` depends on the value of `flavor`:
      - SINGLE_FILE: `factor` must be `1`.
      - SPLIT_FILES: `factor` is the number of partitions per file.
      - FUSED_FILES: `factor` is the number of files per partition.
    """

    __slots__ = ("factor", "flavor", "table_stats")
    factor: int
    flavor: ScanPartitionFlavor
    table_stats: TableStats | None

    def __init__(
        self, factor: int, flavor: ScanPartitionFlavor, table_stats: TableStats | None
    ) -> None:
        if (
            flavor == ScanPartitionFlavor.SINGLE_FILE and factor != 1
        ):  # pragma: no cover
            raise ValueError(f"Expected factor == 1 for {flavor}, got: {factor}")
        self.factor = factor
        self.flavor = flavor
        self.table_stats = table_stats

    @staticmethod
    def from_scan(ir: Scan) -> ScanPartitionPlan:
        """Extract the partitioning plan of a Scan operation."""
        if ir.typ == "parquet":
            # TODO: Use system info to set default blocksize
            assert ir.config_options.executor.name == "streaming", (
                "'in-memory' executor not supported in 'generate_ir_tasks'"
            )

            blocksize: int = ir.config_options.executor.target_partition_size
            table_stats = _sample_pq_statistics(ir)
            file_size = sum(
                cs.file_size
                for name, cs in table_stats.column_stats.items()
                if name in ir.schema and cs.file_size is not None
            )
            if file_size > 0:
                if file_size > blocksize:
                    # Split large files
                    return ScanPartitionPlan(
                        math.ceil(file_size / blocksize),
                        ScanPartitionFlavor.SPLIT_FILES,
                        table_stats,
                    )
                else:
                    # Fuse small files
                    return ScanPartitionPlan(
                        max(blocksize // int(file_size), 1),
                        ScanPartitionFlavor.FUSED_FILES,
                        table_stats,
                    )

        # TODO: Use file sizes for csv and json
        return ScanPartitionPlan(1, ScanPartitionFlavor.SINGLE_FILE, None)


class SplitScan(IR):
    """
    Input from a split file.

    This class wraps a single-file `Scan` object. At
    IO/evaluation time, this class will only perform
    a partial read of the underlying file. The range
    (skip_rows and n_rows) is calculated at IO time.
    """

    __slots__ = (
        "base_scan",
        "schema",
        "split_index",
        "total_splits",
    )
    _non_child = (
        "schema",
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

    def __init__(
        self, schema: Schema, base_scan: Scan, split_index: int, total_splits: int
    ):
        self.schema = schema
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

    @classmethod
    def do_evaluate(
        cls,
        split_index: int,
        total_splits: int,
        schema: Schema,
        typ: str,
        reader_options: dict[str, Any],
        config_options: ConfigOptions,
        paths: list[str],
        with_columns: list[str] | None,
        skip_rows: int,
        n_rows: int,
        row_index: tuple[str, int] | None,
        include_file_paths: str | None,
        predicate: NamedExpr | None,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        if typ not in ("parquet",):  # pragma: no cover
            raise NotImplementedError(f"Unhandled Scan type for file splitting: {typ}")

        if len(paths) > 1:  # pragma: no cover
            raise ValueError(f"Expected a single path, got: {paths}")

        # Parquet logic:
        # - We are one of "total_splits" SplitScan nodes
        #   assigned to the same file.
        # - We know our index within this file ("split_index")
        # - We can also use parquet metadata to query the
        #   total number of rows in each row-group of the file.
        # - We can use all this information to calculate the
        #   "skip_rows" and "n_rows" options to use locally.

        rowgroup_metadata = plc.io.parquet_metadata.read_parquet_metadata(
            plc.io.SourceInfo(paths)
        ).rowgroup_metadata()
        total_row_groups = len(rowgroup_metadata)
        if total_splits <= total_row_groups:
            # We have enough row-groups in the file to align
            # all "total_splits" of our reads with row-group
            # boundaries. Calculate which row-groups to include
            # in the current read, and use metadata to translate
            # the row-group indices to "skip_rows" and "n_rows".
            rg_stride = total_row_groups // total_splits
            skip_rgs = rg_stride * split_index
            skip_rows = sum(rg["num_rows"] for rg in rowgroup_metadata[:skip_rgs])
            n_rows = sum(
                rg["num_rows"]
                for rg in rowgroup_metadata[skip_rgs : skip_rgs + rg_stride]
            )
        else:
            # There are not enough row-groups to align
            # all "total_splits" of our reads with row-group
            # boundaries. Use metadata to directly calculate
            # "skip_rows" and "n_rows" for the current read.
            total_rows = sum(rg["num_rows"] for rg in rowgroup_metadata)
            n_rows = total_rows // total_splits
            skip_rows = n_rows * split_index

        # Last split should always read to end of file
        if split_index == (total_splits - 1):
            n_rows = -1

        # Perform the partial read
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
            include_file_paths,
            predicate,
        )


def _sample_pq_statistics(ir: Scan) -> TableStats:
    # Use average total_uncompressed_size of three files
    n_sample = 5  # TODO: Make this configurable
    file_count = len(ir.paths)
    stride = max(1, int(file_count / n_sample))
    paths = ir.paths[: stride * n_sample : stride]

    # Check table-stats cache
    table_stats_cached: TableStats | None = None
    table_stats_cached_schema: Schema = {}
    try:
        table_stats_cached = _TABLESTATS_CACHE[tuple(ir.paths)]
        table_stats_cached_schema = {
            name: stats.dtype for name, stats in table_stats_cached.column_stats.items()
        }
        table_stats = table_stats_cached
    except KeyError:
        pass

    if need_schema := {
        name: dtype
        for name, dtype in ir.schema.items()
        if name not in table_stats_cached_schema
    }:
        # Still need un-cached columns
        metadata = plc.io.parquet_metadata.read_parquet_metadata(
            plc.io.SourceInfo(paths)
        )
        num_rows_per_file = int(metadata.num_rows() / len(paths))
        num_rows_total = num_rows_per_file * file_count
        num_row_groups_per_file_samples = metadata.num_rowgroups_per_file()
        num_row_groups_per_file = np.mean(num_row_groups_per_file_samples)
        rowgroup_offsets_per_file = np.insert(
            np.cumsum(num_row_groups_per_file_samples), 0, 0
        )

        # For each column, calculate the `total_uncompressed_size` for each file
        column_sizes = {}
        for name, uncompressed_sizes in metadata.columnchunk_metadata().items():
            if name in need_schema:
                column_sizes[name] = np.array(
                    [
                        np.sum(uncompressed_sizes[start:end])
                        for (start, end) in itertools.pairwise(
                            rowgroup_offsets_per_file
                        )
                    ],
                    dtype="int64",
                )
        # Revise schema, since some columns may not be in the file
        need_schema = {k: v for k, v in need_schema.items() if k in column_sizes}

    if need_schema:
        # We have un-cached column metadata to process

        # We already know the element size of most dtypes
        known_element_sizes: dict[str, int] = {
            name: plc.types.size_of(dtype)
            for name, dtype in need_schema.items()
            if dtype.id()
            not in (
                plc.TypeId.LIST,
                plc.TypeId.STRING,
                plc.TypeId.STRUCT,
            )
        }

        # Calculate the mean per-file `total_uncompressed_size` for each column
        total_uncompressed_size = {
            name: np.mean(sizes) for name, sizes in column_sizes.items()
        }
        element_sizes = {
            name: known_element_sizes.get(
                name,
                max(int(total_uncompressed_size[name] / num_rows_per_file), 1),
            )
            for name in need_schema
        }

        # Collect real unique-count of first row-group
        # TODO: Check for 'distinct_count' statistics
        # and make this sampling configurable.
        options = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(paths[:1])
        ).build()
        options.set_columns([c for c in ir.schema if c in need_schema])
        options.set_row_groups([[0]])
        tbl_w_meta = plc.io.parquet.read_parquet(options)
        row_group_num_rows = tbl_w_meta.tbl.num_rows()
        unique_count_estimates: dict[str, int] = {}
        for name, column in zip(
            tbl_w_meta.column_names(), tbl_w_meta.columns, strict=True
        ):
            if name in need_schema:
                row_group_unique_count = plc.stream_compaction.distinct_count(
                    column,
                    plc.types.NullPolicy.INCLUDE,
                    plc.types.NanPolicy.NAN_IS_NULL,
                )
                unique_count_estimates[name] = int(
                    (row_group_unique_count / row_group_num_rows)
                    * row_group_num_rows
                    * np.mean(num_row_groups_per_file)
                    * file_count
                )

        # Construct estimated TableStats
        table_stats = TableStats(
            column_stats={
                name: ColumnStats(
                    dtype=dtype,
                    unique_count=unique_count_estimates[name],
                    element_size=element_sizes[name],
                    file_size=total_uncompressed_size[name],
                )
                for name, dtype in need_schema.items()
            },
            num_rows=num_rows_total,
        )

        if table_stats_cached is not None:
            # Combine new and cached column stats
            table_stats = TableStats.merge(table_stats, table_stats_cached)

    _TABLESTATS_CACHE[tuple(ir.paths)] = table_stats

    return table_stats


@lower_ir_node.register(Scan)
def _(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    partition_info: MutableMapping[IR, PartitionInfo]
    if ir.typ in ("csv", "parquet", "ndjson") and ir.n_rows == -1 and ir.skip_rows == 0:
        plan = ScanPartitionPlan.from_scan(ir)
        paths = list(ir.paths)
        if plan.flavor == ScanPartitionFlavor.SPLIT_FILES:
            # Disable chunked reader when splitting files
            config_options = dataclasses.replace(
                ir.config_options,
                parquet_options=dataclasses.replace(
                    ir.config_options.parquet_options, chunked=False
                ),
            )

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
                    ir.include_file_paths,
                    ir.predicate,
                )
                slices.extend(
                    SplitScan(ir.schema, base_scan, sindex, plan.factor)
                    for sindex in range(plan.factor)
                )
            new_node = Union(ir.schema, None, *slices)
            partition_info = {slice: PartitionInfo(count=1) for slice in slices} | {
                new_node: PartitionInfo(count=len(slices), table_stats=plan.table_stats)
            }
        else:
            groups: list[Scan] = [
                Scan(
                    ir.schema,
                    ir.typ,
                    ir.reader_options,
                    ir.cloud_options,
                    ir.config_options,
                    paths[i : i + plan.factor],
                    ir.with_columns,
                    ir.skip_rows,
                    ir.n_rows,
                    ir.row_index,
                    ir.include_file_paths,
                    ir.predicate,
                )
                for i in range(0, len(paths), plan.factor)
            ]
            new_node = Union(ir.schema, None, *groups)
            partition_info = {group: PartitionInfo(count=1) for group in groups} | {
                new_node: PartitionInfo(count=len(groups), table_stats=plan.table_stats)
            }
        return new_node, partition_info

    return ir, {ir: PartitionInfo(count=1)}  # pragma: no cover
