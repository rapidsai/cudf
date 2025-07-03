# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition IO Logic."""

from __future__ import annotations

import dataclasses
import enum
import functools
import itertools
import math
import statistics
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

import pylibcudf as plc

from cudf_polars.dsl.ir import IR, DataFrameScan, Scan, Sink, Union
from cudf_polars.experimental.base import (
    ColumnStats,
    DataSourceInfo,
    PartitionInfo,
    RowCountInfo,
    StorageSizeInfo,
    UniqueInfo,
    get_key_name,
)
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions, ParquetOptions


@lower_ir_node.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    config_options = rec.state["config_options"]

    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_tasks'"
    )

    rows_per_partition = config_options.executor.max_rows_per_partition
    nrows = max(ir.df.shape()[0], 1)
    count = math.ceil(nrows / rows_per_partition)

    if count > 1:
        length = math.ceil(nrows / count)
        slices = [
            DataFrameScan(
                ir.schema,
                ir.df.slice(offset, length),
                ir.projection,
            )
            for offset in range(0, nrows, length)
        ]
        new_node = Union(ir.schema, None, *slices)
        return new_node, {slice: PartitionInfo(count=1) for slice in slices} | {
            new_node: PartitionInfo(count=count)
        }

    return ir, {ir: PartitionInfo(count=1)}


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

    __slots__ = ("factor", "flavor")
    factor: int
    flavor: ScanPartitionFlavor

    def __init__(self, factor: int, flavor: ScanPartitionFlavor) -> None:
        if (
            flavor == ScanPartitionFlavor.SINGLE_FILE and factor != 1
        ):  # pragma: no cover
            raise ValueError(f"Expected factor == 1 for {flavor}, got: {factor}")
        self.factor = factor
        self.flavor = flavor

    @staticmethod
    def from_scan(ir: Scan, config_options: ConfigOptions) -> ScanPartitionPlan:
        """Extract the partitioning plan of a Scan operation."""
        if ir.typ == "parquet":
            # TODO: Use system info to set default blocksize
            assert config_options.executor.name == "streaming", (
                "'in-memory' executor not supported in 'generate_ir_tasks'"
            )

            blocksize: int = config_options.executor.target_partition_size
            column_stats = _extract_scan_stats(ir)
            column_sizes: list[int] = []
            for name, cs in column_stats.items():
                storage_size = cs.source.storage_size(name)
                if storage_size.value is not None:
                    column_sizes.append(storage_size.value)

            if (file_size := sum(column_sizes)) > 0:
                if file_size > blocksize:
                    # Split large files
                    return ScanPartitionPlan(
                        math.ceil(file_size / blocksize),
                        ScanPartitionFlavor.SPLIT_FILES,
                    )
                else:
                    # Fuse small files
                    return ScanPartitionPlan(
                        max(blocksize // int(file_size), 1),
                        ScanPartitionFlavor.FUSED_FILES,
                    )

        # TODO: Use file sizes for csv and json
        return ScanPartitionPlan(1, ScanPartitionFlavor.SINGLE_FILE)


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
        "parquet_options",
        "schema",
        "split_index",
        "total_splits",
    )
    _non_child = (
        "schema",
        "base_scan",
        "split_index",
        "total_splits",
        "parquet_options",
    )
    base_scan: Scan
    """Scan operation this node is based on."""
    split_index: int
    """Index of the current split."""
    total_splits: int
    """Total number of splits."""
    parquet_options: ParquetOptions
    """Parquet-specific options."""

    def __init__(
        self,
        schema: Schema,
        base_scan: Scan,
        split_index: int,
        total_splits: int,
        parquet_options: ParquetOptions,
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
        self.parquet_options = parquet_options
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
        paths: list[str],
        with_columns: list[str] | None,
        skip_rows: int,
        n_rows: int,
        row_index: tuple[str, int] | None,
        include_file_paths: str | None,
        predicate: NamedExpr | None,
        parquet_options: ParquetOptions,
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
            paths,
            with_columns,
            skip_rows,
            n_rows,
            row_index,
            include_file_paths,
            predicate,
            parquet_options,
        )


@lower_ir_node.register(Scan)
def _(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    partition_info: MutableMapping[IR, PartitionInfo]
    config_options = rec.state["config_options"]
    if ir.typ in ("csv", "parquet", "ndjson") and ir.n_rows == -1 and ir.skip_rows == 0:
        plan = ScanPartitionPlan.from_scan(ir, config_options)
        paths = list(ir.paths)
        if plan.flavor == ScanPartitionFlavor.SPLIT_FILES:
            # Disable chunked reader when splitting files
            parquet_options = dataclasses.replace(
                config_options.parquet_options,
                chunked=False,
            )

            slices: list[SplitScan] = []
            for path in paths:
                base_scan = Scan(
                    ir.schema,
                    ir.typ,
                    ir.reader_options,
                    ir.cloud_options,
                    [path],
                    ir.with_columns,
                    ir.skip_rows,
                    ir.n_rows,
                    ir.row_index,
                    ir.include_file_paths,
                    ir.predicate,
                    parquet_options,
                )
                slices.extend(
                    SplitScan(
                        ir.schema, base_scan, sindex, plan.factor, parquet_options
                    )
                    for sindex in range(plan.factor)
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
                    paths[i : i + plan.factor],
                    ir.with_columns,
                    ir.skip_rows,
                    ir.n_rows,
                    ir.row_index,
                    ir.include_file_paths,
                    ir.predicate,
                    config_options.parquet_options,
                )
                for i in range(0, len(paths), plan.factor)
            ]
            new_node = Union(ir.schema, None, *groups)
            partition_info = {group: PartitionInfo(count=1) for group in groups} | {
                new_node: PartitionInfo(count=len(groups))
            }
        return new_node, partition_info

    return ir, {ir: PartitionInfo(count=1)}  # pragma: no cover


@lower_ir_node.register(Sink)
def _(
    ir: Sink, rec: LowerIRTransformer
) -> tuple[Sink, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])
    if Path(ir.path).exists():
        # TODO: Support cloud storage
        raise NotImplementedError(
            "Writing to an existing path is not supported "
            "by the GPU streaming executor."
        )
    new_node = ir.reconstruct([child])
    partition_info[new_node] = partition_info[child]
    return new_node, partition_info


def _prepare_sink(path: str) -> None:
    """Prepare for a multi-partition sink."""
    # TODO: Support cloud storage
    Path(path).mkdir(parents=True)


def _sink_partition(
    schema: Schema,
    kind: str,
    path: str,
    options: dict[str, Any],
    df: DataFrame,
    ready: None,
) -> DataFrame:
    """Sink a partition to disk."""
    return Sink.do_evaluate(schema, kind, path, options, df)


@generate_ir_tasks.register(Sink)
def _(
    ir: Sink, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    name = get_key_name(ir)
    count = partition_info[ir].count
    child_name = get_key_name(ir.children[0])
    if count == 1:
        return {
            (name, 0): (
                ir.do_evaluate,
                *ir._non_child_args,
                (child_name, 0),
            )
        }

    setup_name = f"setup-{name}"
    suffix = ir.kind.lower()
    width = math.ceil(math.log10(count))
    graph: MutableMapping[Any, Any] = {
        (name, i): (
            _sink_partition,
            ir.schema,
            ir.kind,
            f"{ir.path}/part.{str(i).zfill(width)}.{suffix}",
            ir.options,
            (child_name, i),
            setup_name,
        )
        for i in range(count)
    }
    graph[setup_name] = (_prepare_sink, ir.path)
    return graph


class PqSourceInfo(DataSourceInfo):
    """
    Parquet datasource information.

    Parameters
    ----------
    paths
        Sample paths.
    max_file_samples
        Maximum number of files to sample metadata from.
    max_rg_samples
        Maximum number of row-groups to sample data from.
    """

    def __init__(
        self,
        paths: tuple[str, ...],
        max_file_samples: int,
        max_rg_samples: int,
    ):
        self.paths = paths
        self.max_file_samples = max_file_samples
        self.max_rg_samples = max_rg_samples
        # Helper attributes
        self._key_columns: set[str] = set()
        self._unique_stats: dict[str, UniqueInfo] = {}
        self._row_count: RowCountInfo | None = None
        self._num_row_groups_per_file: Sequence[int] | None = None
        self._mean_size_per_file: dict[str, StorageSizeInfo] | None = None
        self._sample_paths: tuple[str, ...] | None = None

    def _sample_metadata(self) -> None:
        """Sample Parquet metadata."""
        total_file_count = len(self.paths)
        stride = max(1, int(total_file_count / self.max_file_samples))
        self._sample_paths = self.paths[: stride * self.max_file_samples : stride]
        sampled_file_count = len(self._sample_paths)
        exact: bool = False

        sample_metadata = plc.io.parquet_metadata.read_parquet_metadata(
            plc.io.SourceInfo(list(self._sample_paths))
        )

        if total_file_count == sampled_file_count:
            # We know the "exact" row_count from our sample
            row_count = sample_metadata.num_rows()
            exact = True
        else:
            # We must estimate/extrapolate the row_count from our sample
            num_rows_per_sampled_file = int(
                sample_metadata.num_rows() / sampled_file_count
            )
            row_count = num_rows_per_sampled_file * total_file_count

        num_row_groups_per_sampled_file = sample_metadata.num_rowgroups_per_file()
        rowgroup_offsets_per_file = list(
            itertools.accumulate(num_row_groups_per_sampled_file, initial=0)
        )

        column_sizes_per_file = {
            name: [
                sum(uncompressed_sizes[start:end])
                for (start, end) in itertools.pairwise(rowgroup_offsets_per_file)
            ]
            for name, uncompressed_sizes in sample_metadata.columnchunk_metadata().items()
        }

        self._mean_size_per_file = {
            name: StorageSizeInfo(value=int(statistics.mean(sizes)))
            for name, sizes in column_sizes_per_file.items()
        }
        self._num_row_groups_per_file = num_row_groups_per_sampled_file
        self._row_count = RowCountInfo(value=row_count, exact=exact)

    def _sample_row_groups(self) -> None:
        """Estimate unique-value statistics from a row-group sample."""
        if self._row_count is None:
            self._sample_metadata()  # pragma: no cover; Usually sampled before row-groups
        if (
            self._num_row_groups_per_file is None
            or self._sample_paths is None
            or self._row_count is None
        ):
            raise ValueError("Parquet metadata sampling failed.")  # pragma: no cover

        n = 0
        samples: defaultdict[str, list[int]] = defaultdict(list)
        for path, num_rgs in zip(
            self._sample_paths, self._num_row_groups_per_file, strict=True
        ):
            for rg_id in range(num_rgs):
                n += 1
                samples[path].append(rg_id)
                if n == self.max_rg_samples:
                    break
            if n == self.max_rg_samples:
                break

        exact = len(self._sample_paths) == len(
            self.paths
        ) and self.max_rg_samples >= sum(self._num_row_groups_per_file)

        options = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(list(samples))
        ).build()
        options.set_columns(list(self._key_columns))
        options.set_row_groups(list(samples.values()))
        tbl_w_meta = plc.io.parquet.read_parquet(options)
        row_group_num_rows = tbl_w_meta.tbl.num_rows()
        for name, column in zip(
            tbl_w_meta.column_names(), tbl_w_meta.columns, strict=True
        ):
            row_group_unique_count = plc.stream_compaction.distinct_count(
                column,
                plc.types.NullPolicy.INCLUDE,
                plc.types.NanPolicy.NAN_IS_NULL,
            )
            result = {
                "exact": exact,
                "fraction": row_group_unique_count / row_group_num_rows,
            }
            # Assume that if every row is unique then this is a
            # primary key otherwise it's a foreign key and we
            # can't use the single row group count estimate
            # Example, consider a "foreign" key that has 100
            # unique values. If we sample from a single row group,
            # we likely obtain a unique count of 100. But we can't
            # necessarily deduce that that means that the unique
            # count is 100 / num_rows_in_group * num_rows_in_file
            if exact:
                result["count"] = row_group_unique_count
            elif row_group_unique_count == row_group_num_rows:
                result["count"] = self._row_count.value

            self._unique_stats[name] = UniqueInfo(**result)

    @property
    def row_count(self) -> RowCountInfo:
        """Data source row-count estimate."""
        if self.max_file_samples < 1:
            return RowCountInfo()  # pragma: no cover

        if self._row_count is None:
            self._sample_metadata()

        assert self._row_count is not None, "metadata sampling failed."
        return self._row_count

    def unique(self, column: str) -> UniqueInfo:
        """Return unique-value information."""
        if self.max_file_samples < 1 or self.max_rg_samples < 1:
            return UniqueInfo()  # pragma: no cover

        if column not in self._unique_stats:
            self.add_unique_stats_column(column)
            self._sample_row_groups()
            self._key_columns = set()

        return self._unique_stats.get(column, UniqueInfo())

    def storage_size(self, column: str) -> StorageSizeInfo:
        """Return the average column size for a single file."""
        if self.max_file_samples < 1:
            return StorageSizeInfo()  # pragma: no cover

        if self._mean_size_per_file is None:
            self._sample_metadata()

        assert self._mean_size_per_file is not None, "metadata sampling failed"
        return self._mean_size_per_file.get(column, StorageSizeInfo())

    def add_unique_stats_column(self, column: str) -> None:
        """Add a column needing unique-value information."""
        if column not in self._key_columns and column not in self._unique_stats:
            self._key_columns.add(column)


@functools.lru_cache(maxsize=10)
def _sample_pq_stats(
    paths: tuple[str, ...],
    max_file_samples: int,
    max_rg_samples: int,
) -> PqSourceInfo:
    """Return Parquet datasource information."""
    return PqSourceInfo(paths, max_file_samples, max_rg_samples)


def _extract_scan_stats(ir: Scan) -> dict[str, ColumnStats]:
    """Extract base ColumnStats for a Scan node."""
    if ir.typ == "parquet":
        # TODO: Make max_file_samples and max_rg_samples configurable
        max_file_samples = 3
        max_rg_samples = 1
        source_stats = _sample_pq_stats(
            tuple(ir.paths), max_file_samples, max_rg_samples
        )
        return {
            name: ColumnStats(
                name=name,
                source=source_stats,
                source_name=name,
            )
            for name in ir.schema
        }

    else:
        return {name: ColumnStats(name=name) for name in ir.schema}


class DataFrameSourceInfo(DataSourceInfo):
    """
    In-memory DataFrame source information.

    Parameters
    ----------
    df
        In-memory DataFrame source.
    """

    def __init__(self, df: Any):
        self._df = df
        self._key_columns: set[str] = set()
        self._unique_stats: dict[str, UniqueInfo] = {}

    @functools.cached_property
    def row_count(self) -> RowCountInfo:
        """Data source row-count estimate."""
        return RowCountInfo(value=self._df.height(), exact=True)

    def unique(self, column: str) -> UniqueInfo:
        """Return unique-value information."""
        if column not in self._unique_stats:
            row_count = self.row_count.value
            count = (
                pl.DataFrame._from_pydf(self._df).n_unique(subset=[column])
                if row_count
                else 0
            )
            fraction = (count / row_count) if row_count else 1.0
            self._unique_stats[column] = UniqueInfo(
                count=count,
                fraction=fraction,
                exact=True,
            )
        return self._unique_stats[column]


def _extract_dataframescan_stats(ir: DataFrameScan) -> dict[str, ColumnStats]:
    """Extract base ColumnStats for a DataFrameScan node."""
    source_stats = DataFrameSourceInfo(ir.df)
    return {
        name: ColumnStats(
            name=name,
            source=source_stats,
            source_name=name,
        )
        for name in ir.schema
    }
