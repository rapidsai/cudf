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

import pylibcudf as plc

from cudf_polars.dsl.ir import IR, DataFrameScan, Scan, Sink, Union
from cudf_polars.experimental.base import (
    ColumnStat,
    ColumnStats,
    DataSourceInfo,
    PartitionInfo,
    UniqueStats,
    get_key_name,
)
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node

if TYPE_CHECKING:
    from collections.abc import Hashable, MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import (
        ConfigOptions,
        ParquetOptions,
        StreamingExecutor,
    )


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
            column_stats = _extract_scan_stats(ir, config_options)
            column_sizes: list[int] = []
            for name, cs in column_stats.items():
                storage_size = cs.source_info.storage_size(name)
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
    if (
        ir.typ in ("csv", "parquet", "ndjson")
        and ir.n_rows == -1
        and ir.skip_rows == 0
        and ir.row_index is None
    ):
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


class StreamingSink(IR):
    """Sink a dataframe in streaming mode."""

    __slots__ = ("executor_options", "sink")
    _non_child = ("schema", "sink", "executor_options")

    sink: Sink
    executor_options: StreamingExecutor

    def __init__(
        self,
        schema: Schema,
        sink: Sink,
        executor_options: StreamingExecutor,
        df: IR,
    ):
        self.schema = schema
        self.sink = sink
        self.executor_options = executor_options
        self.children = (df,)

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        return (type(self), self.sink, *self.children)


@lower_ir_node.register(Sink)
def _(
    ir: Sink, rec: LowerIRTransformer
) -> tuple[StreamingSink, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])
    executor_options = rec.state["config_options"].executor

    assert executor_options.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node'"
    )

    # TODO: Support cloud storage
    if Path(ir.path).exists() and executor_options.sink_to_directory:
        raise NotImplementedError(
            "Writing to an existing path is not supported when sinking "
            "to a directory. If you are using the 'distributed' scheduler, "
            "please remove the target directory before calling 'collect'. "
        )

    new_node = StreamingSink(
        ir.schema,
        ir.reconstruct([child]),
        executor_options,
        child,
    )
    partition_info[new_node] = partition_info[child]
    return new_node, partition_info


def _prepare_sink_directory(path: str) -> None:
    """Prepare for a multi-partition sink."""
    # TODO: Support cloud storage
    Path(path).mkdir(parents=True)


def _sink_to_directory(
    schema: Schema,
    kind: str,
    path: str,
    options: dict[str, Any],
    df: DataFrame,
    ready: None,
) -> DataFrame:
    """Sink a partition to a new file."""
    return Sink.do_evaluate(schema, kind, path, options, df)


def _sink_to_parquet_file(
    path: str,
    options: dict[str, Any],
    finalize: bool,  # noqa: FBT001
    writer: plc.io.parquet.ChunkedParquetWriter | None,
    df: DataFrame,
) -> plc.io.parquet.ChunkedParquetWriter | DataFrame:
    """Sink a partition to an open Parquet file."""
    # Set up a new chunked Parquet writer if necessary.
    if writer is None:
        metadata = plc.io.types.TableInputMetadata(df.table)
        for i, name in enumerate(df.column_names):
            metadata.column_metadata[i].set_name(name)

        sink = plc.io.types.SinkInfo([path])
        builder = plc.io.parquet.ChunkedParquetWriterOptions.builder(sink)
        compression = options["compression"]
        if compression != "Uncompressed":
            builder.compression(
                getattr(plc.io.types.CompressionType, compression.upper())
            )

        if options["data_page_size"] is not None:
            builder.max_page_size_bytes(options["data_page_size"])
        if options["row_group_size"] is not None:
            builder.row_group_size_rows(options["row_group_size"])

        writer_options = builder.metadata(metadata).build()
        writer = plc.io.parquet.ChunkedParquetWriter.from_options(writer_options)

    # Append to the open Parquet file.
    assert isinstance(writer, plc.io.parquet.ChunkedParquetWriter), (
        "ChunkedParquetWriter is required."
    )
    writer.write(df.table)

    # Finalize or return active writer.
    if finalize:
        writer.close([])
        return df
    else:
        return writer


def _sink_to_file(
    kind: str,
    path: str,
    options: dict[str, Any],
    finalize: bool,  # noqa: FBT001
    writer_state: Any,
    df: DataFrame,
) -> Any:
    """Sink a partition to an open file."""
    if kind == "Parquet":
        # Parquet writer will pass along a
        # ChunkedParquetWriter "writer state".
        return _sink_to_parquet_file(
            path,
            options,
            finalize,
            writer_state,
            df,
        )
    elif kind == "Csv":
        use_options = options.copy()
        if writer_state is None:
            mode = "wb"
        else:
            mode = "ab"
            use_options["include_header"] = False
        with Path.open(Path(path), mode) as f:
            sink = plc.io.types.SinkInfo([f])
            Sink._write_csv(sink, use_options, df)
    elif kind == "Json":
        mode = "wb" if writer_state is None else "ab"
        with Path.open(Path(path), mode) as f:
            sink = plc.io.types.SinkInfo([f])
            Sink._write_json(sink, df)
    else:  # pragma: no cover; Shouldn't get here.
        raise NotImplementedError(f"{kind} not yet supported in _sink_to_file")

    # Default return type is bool | DataFrame.
    # We only return a DataFrame for the final sink task.
    # The other tasks return a "ready" signal of True.
    return df if finalize else True


def _file_sink_graph(
    ir: StreamingSink, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    """Sink to a single file."""
    name = get_key_name(ir)
    count = partition_info[ir].count
    child_name = get_key_name(ir.children[0])
    sink = ir.sink
    if count == 1:
        return {
            (name, 0): (
                sink.do_evaluate,
                *sink._non_child_args,
                (child_name, 0),
            )
        }

    sink_name = get_key_name(sink)
    graph: MutableMapping[Any, Any] = {
        (sink_name, i): (
            _sink_to_file,
            sink.kind,
            sink.path,
            sink.options,
            i == count - 1,  # Whether to finalize
            None if i == 0 else (sink_name, i - 1),  # Writer state
            (child_name, i),
        )
        for i in range(count)
    }

    # Make sure final tasks point to empty DataFrame output
    graph.update({(name, i): (sink_name, count - 1) for i in range(count)})
    return graph


def _directory_sink_graph(
    ir: StreamingSink, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    """Sink to a directory of files."""
    name = get_key_name(ir)
    count = partition_info[ir].count
    child_name = get_key_name(ir.children[0])
    sink = ir.sink

    setup_name = f"setup-{name}"
    suffix = sink.kind.lower()
    width = math.ceil(math.log10(count))
    graph: MutableMapping[Any, Any] = {
        (name, i): (
            _sink_to_directory,
            sink.schema,
            sink.kind,
            f"{sink.path}/part.{str(i).zfill(width)}.{suffix}",
            sink.options,
            (child_name, i),
            setup_name,
        )
        for i in range(count)
    }
    graph[setup_name] = (_prepare_sink_directory, sink.path)
    return graph


@generate_ir_tasks.register(StreamingSink)
def _(
    ir: StreamingSink, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    if ir.executor_options.sink_to_directory:
        return _directory_sink_graph(ir, partition_info)
    else:
        return _file_sink_graph(ir, partition_info)


class ParquetMetadata:
    """
    Parquet metadata container.

    Parameters
    ----------
    paths
        Parquet-dataset paths.
    max_footer_samples
        Maximum number of file footers to sample metadata from.
    """

    __slots__ = (
        "column_names",
        "max_footer_samples",
        "mean_size_per_file",
        "num_row_groups_per_file",
        "paths",
        "row_count",
        "sample_paths",
    )

    paths: tuple[str, ...]
    """Parquet-dataset paths."""
    max_footer_samples: int
    """Maximum number of file footers to sample metadata from."""
    row_count: ColumnStat[int]
    """Total row-count estimate."""
    num_row_groups_per_file: tuple[int, ...]
    """Number of row groups in each sampled file."""
    mean_size_per_file: dict[str, ColumnStat[int]]
    """Average column storage size in a single file."""
    column_names: tuple[str, ...]
    """All column names found it the dataset."""
    sample_paths: tuple[str, ...]
    """Sampled file paths."""

    def __init__(self, paths: tuple[str, ...], max_footer_samples: int):
        self.paths = paths
        self.max_footer_samples = max_footer_samples
        self.row_count = ColumnStat[int]()
        self.num_row_groups_per_file = ()
        self.mean_size_per_file = {}
        self.column_names = ()
        stride = (
            max(1, int(len(paths) / max_footer_samples)) if max_footer_samples else 1
        )
        self.sample_paths = paths[: stride * max_footer_samples : stride]

        if not self.sample_paths:
            # No paths to sample from
            return

        total_file_count = len(self.paths)
        sampled_file_count = len(self.sample_paths)
        exact: bool = False
        sample_metadata = plc.io.parquet_metadata.read_parquet_metadata(
            plc.io.SourceInfo(list(self.sample_paths))
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

        self.column_names = tuple(column_sizes_per_file)
        self.mean_size_per_file = {
            name: ColumnStat[int](value=int(statistics.mean(sizes)))
            for name, sizes in column_sizes_per_file.items()
        }
        self.num_row_groups_per_file = tuple(num_row_groups_per_sampled_file)
        self.row_count.value = row_count
        self.row_count.exact = exact


class ParquetSourceInfo(DataSourceInfo):
    """
    Parquet datasource information.

    Parameters
    ----------
    paths
        Parquet-dataset paths.
    max_footer_samples
        Maximum number of file footers to sample metadata from.
    max_row_group_samples
        Maximum number of row-groups to sample data from.
    """

    def __init__(
        self,
        paths: tuple[str, ...],
        max_footer_samples: int,
        max_row_group_samples: int,
    ):
        self.paths = paths
        self.max_footer_samples = max_footer_samples
        self.max_row_group_samples = max_row_group_samples
        # Helper attributes
        self._key_columns: set[str] = set()  # Used to fuse lazy row-group sampling
        self._unique_stats: dict[str, UniqueStats] = {}

    @functools.cached_property
    def metadata(self) -> ParquetMetadata:
        """Return Parquet metadata."""
        return ParquetMetadata(self.paths, self.max_footer_samples)

    @property
    def row_count(self) -> ColumnStat[int]:
        """Data source row-count estimate."""
        return self.metadata.row_count

    def _sample_row_groups(self) -> None:
        """Estimate unique-value statistics from a row-group sample."""
        sample_paths = self.metadata.sample_paths
        if not sample_paths or self.max_row_group_samples < 1:
            # No row-groups to sample from
            return

        column_names = self.metadata.column_names
        if not (
            key_columns := [key for key in self._key_columns if key in column_names]
        ):  # pragma: no cover; should never get here
            # No key columns found in the file
            raise ValueError(f"None of {self._key_columns} in {column_names}")

        sampled_file_count = len(sample_paths)
        num_row_groups_per_file = self.metadata.num_row_groups_per_file
        if (
            self.row_count.value is None
            or len(num_row_groups_per_file) != sampled_file_count
        ):
            raise ValueError("Parquet metadata sampling failed.")  # pragma: no cover

        n = 0
        samples: defaultdict[str, list[int]] = defaultdict(list)
        for path, num_rgs in zip(sample_paths, num_row_groups_per_file, strict=True):
            for rg_id in range(num_rgs):
                n += 1
                samples[path].append(rg_id)
                if n == self.max_row_group_samples:
                    break
            if n == self.max_row_group_samples:
                break

        exact = sampled_file_count == len(
            self.paths
        ) and self.max_row_group_samples >= sum(num_row_groups_per_file)

        options = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(list(samples))
        ).build()
        options.set_columns(key_columns)
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
            fraction = row_group_unique_count / row_group_num_rows
            # Assume that if every row is unique then this is a
            # primary key otherwise it's a foreign key and we
            # can't use the single row group count estimate.
            # Example, consider a "foreign" key that has 100
            # unique values. If we sample from a single row group,
            # we likely obtain a unique count of 100. But we can't
            # necessarily deduce that that means that the unique
            # count is 100 / num_rows_in_group * num_rows_in_file
            count: int | None = None
            if exact:
                count = row_group_unique_count
            elif row_group_unique_count == row_group_num_rows:
                count = self.row_count.value
            self._unique_stats[name] = UniqueStats(
                ColumnStat[int](value=count, exact=exact),
                ColumnStat[float](value=fraction, exact=exact),
            )

    def _update_unique_stats(self, column: str) -> None:
        if column not in self._unique_stats and column in self.metadata.column_names:
            self.add_unique_stats_column(column)
            self._sample_row_groups()
            self._key_columns = set()

    def unique_stats(self, column: str) -> UniqueStats:
        """Return unique-value statistics for a column."""
        self._update_unique_stats(column)
        return self._unique_stats.get(column, UniqueStats())

    def storage_size(self, column: str) -> ColumnStat[int]:
        """Return the average column size for a single file."""
        return self.metadata.mean_size_per_file.get(column, ColumnStat[int]())

    def add_unique_stats_column(self, column: str) -> None:
        """Add a column needing unique-value information."""
        if column not in self._key_columns and column not in self._unique_stats:
            self._key_columns.add(column)


@functools.cache
def _sample_pq_stats(
    paths: tuple[str, ...],
    max_footer_samples: int,
    max_row_group_samples: int,
) -> ParquetSourceInfo:
    """Return Parquet datasource information."""
    return ParquetSourceInfo(paths, max_footer_samples, max_row_group_samples)


def _extract_scan_stats(
    ir: Scan,
    config_options: ConfigOptions,
) -> dict[str, ColumnStats]:
    """Extract base ColumnStats for a Scan node."""
    if ir.typ == "parquet":
        source_info = _sample_pq_stats(
            tuple(ir.paths),
            config_options.parquet_options.max_footer_samples,
            config_options.parquet_options.max_row_group_samples,
        )
        return {
            name: ColumnStats(
                name=name,
                source_info=source_info,
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
        self._unique_stats: dict[str, UniqueStats] = {}

    @functools.cached_property
    def row_count(self) -> ColumnStat[int]:
        """Data source row-count estimate."""
        return ColumnStat[int](value=self._df.height(), exact=True)

    def _update_unique_stats(self, column: str) -> None:
        if column not in self._unique_stats:
            row_count = self.row_count.value
            unique_count = (
                self._df.get_column(column).approx_n_unique() if row_count else 0
            )
            unique_fraction = min((unique_count / row_count), 1.0) if row_count else 1.0
            self._unique_stats[column] = UniqueStats(
                ColumnStat[int](value=unique_count),
                ColumnStat[float](value=unique_fraction),
            )

    def unique_stats(self, column: str) -> UniqueStats:
        """Return unique-value statistics for a column."""
        self._update_unique_stats(column)
        return self._unique_stats.get(column, UniqueStats())


def _extract_dataframescan_stats(ir: DataFrameScan) -> dict[str, ColumnStats]:
    """Extract base ColumnStats for a DataFrameScan node."""
    source_info = DataFrameSourceInfo(ir.df)
    return {
        name: ColumnStats(
            name=name,
            source_info=source_info,
            source_name=name,
        )
        for name in ir.schema
    }


def _clear_source_info_cache() -> None:
    """Clear DataSourceInfo caches."""
    # TODO: Avoid clearing the cache if we can
    # check that the underlying data hasn't changed.

    # Clear ParquetSourceInfo cache
    _sample_pq_stats.cache_clear()
