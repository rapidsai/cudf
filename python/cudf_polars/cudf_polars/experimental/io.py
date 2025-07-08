# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition IO Logic."""

from __future__ import annotations

import dataclasses
import enum
import itertools
import math
import random
import statistics
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pylibcudf as plc

from cudf_polars.dsl.ir import IR, DataFrameScan, Scan, Sink, Union
from cudf_polars.experimental.base import PartitionInfo, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, MutableMapping

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
            # _sample_pq_statistics is generic over the bit-width of the array
            # We don't care about that here, so we ignore it.
            stats = _sample_pq_statistics(ir)  # type: ignore[var-annotated]
            # Some columns (e.g., "include_file_paths") may be present in the schema
            # but not in the Parquet statistics dict. We use stats.get(column, 0)
            # to safely fall back to 0 in those cases.
            file_size = sum(float(stats.get(column, 0)) for column in ir.schema)
            if file_size > 0:
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


def _sample_pq_statistics(ir: Scan) -> dict[str, float]:
    # Use average total_uncompressed_size of three files
    n_sample = min(3, len(ir.paths))
    metadata = plc.io.parquet_metadata.read_parquet_metadata(
        plc.io.SourceInfo(random.sample(ir.paths, n_sample))
    )
    rowgroup_offsets_per_file = tuple(
        itertools.accumulate(metadata.num_rowgroups_per_file(), initial=0)
    )

    # Return the mean per-file `total_uncompressed_size` for each column
    return {
        name: statistics.mean(
            sum(uncompressed_sizes[start:end])
            for (start, end) in itertools.pairwise(rowgroup_offsets_per_file)
        )
        for name, uncompressed_sizes in metadata.columnchunk_metadata().items()
    }


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
    schema: Schema,
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


def _sink_to_csv_file(
    schema: Schema,
    path: str,
    options: dict[str, Any],
    finalize: bool,  # noqa: FBT001
    ready: bool | None,
    df: DataFrame,
) -> bool | DataFrame:
    """Sink a partition to an open Parquet file."""
    # Write to BytesIO buffer.
    # TODO: Append directly to existing file if/when possible.
    buffer = BytesIO()
    target = plc.io.types.SinkInfo([buffer])
    serialize = options["serialize_options"]
    include_header = options["include_header"] if ready is None else False
    options = (
        plc.io.csv.CsvWriterOptions.builder(target, df.table)
        .include_header(include_header)
        .names(df.column_names if include_header else [])
        .na_rep(serialize["null"])
        .line_terminator(serialize["line_terminator"])
        .inter_column_delimiter(chr(serialize["separator"]))
        .build()
    )
    plc.io.csv.write_csv(options)

    # Append BytesIO buffer to path
    with Path.open(Path(path), "ab") as f:
        buffer.seek(0)
        f.write(buffer.getvalue())

    # Finalize or return ready signal
    return df if finalize else True


def _sink_to_json_file(
    schema: Schema,
    path: str,
    options: dict[str, Any],
    finalize: bool,  # noqa: FBT001
    ready: None,
    df: DataFrame,
) -> DataFrame | None:
    """Sink a partition to an open Json file."""
    # Write to BytesIO buffer.
    # TODO: Append directly to existing file if/when possible.
    buffer = BytesIO()
    target = plc.io.types.SinkInfo([buffer])
    metadata = plc.io.TableWithMetadata(
        df.table, [(col, []) for col in df.column_names]
    )
    options = (
        plc.io.json.JsonWriterOptions.builder(target, df.table)
        .lines(val=True)
        .na_rep("null")
        .include_nulls(val=True)
        .metadata(metadata)
        .utf8_escaped(val=False)
        .build()
    )
    plc.io.json.write_json(options)

    # Append BytesIO buffer to path
    with Path.open(Path(path), "ab") as f:
        buffer.seek(0)
        f.write(buffer.getvalue())

    # Finalize or return ready signal
    return df if finalize else None


def _sink_to_file(
    schema: Schema,
    kind: str,
    path: str,
    options: dict[str, Any],
    finalize: bool,  # noqa: FBT001
    writer_state: Any,
    df: DataFrame,
) -> Any:
    """Sink a partition to an open file."""
    sink_function: Callable[..., Any]
    if kind == "Parquet":
        sink_function = _sink_to_parquet_file
    elif kind == "Csv":
        sink_function = _sink_to_csv_file
    elif kind == "Json":
        sink_function = _sink_to_json_file
    else:  # pragma: no cover; Shouldn't get here.
        raise NotImplementedError(f"{kind} not yet supported in _sink_to_file")

    return sink_function(
        schema,
        path,
        options,
        finalize,
        writer_state,
        df,
    )


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
            sink.schema,
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
