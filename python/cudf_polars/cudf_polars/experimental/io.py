# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition IO Logic."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import statistics
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl

import pylibcudf as plc

from cudf_polars.dsl.ir import (
    IR,
    DataFrameScan,
    Empty,
    Scan,
    Sink,
    Union,
)
from cudf_polars.experimental.base import (
    IOPartitionFlavor,
    IOPartitionPlan,
    PartitionInfo,
    SerializedDataSourceInfo,
    get_key_name,
)
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.utils.config import Cluster
from cudf_polars.utils.cuda_stream import get_cuda_stream
from cudf_polars.utils.versions import POLARS_VERSION_LT_137

if TYPE_CHECKING:
    from collections.abc import Hashable, MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.base import (
        DataSourceInfo,
        SerializedDataSourceInfo,
        StatsCollector,
    )
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

    # RapidsMPF runtime: Use rapidsmpf-specific lowering
    if (
        config_options.executor.runtime == "rapidsmpf"
    ):  # pragma: no cover; Requires rapidsmpf runtime
        from cudf_polars.experimental.rapidsmpf.io import lower_dataframescan_rapidsmpf

        return lower_dataframescan_rapidsmpf(ir, rec)

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


def scan_partition_plan(
    ir: Scan, stats: StatsCollector, config_options: ConfigOptions[StreamingExecutor]
) -> IOPartitionPlan:
    """Extract the partitioning plan of a Scan operation."""
    if ir.typ == "parquet":
        blocksize: int = config_options.executor.target_partition_size
        if source := stats.scan_stats.get(ir):
            column_sizes = [
                sz
                for col in ir.schema
                if (sz := source.column_storage_size(col)) is not None
            ]
            if (file_size := sum(column_sizes)) > 0:
                if file_size > blocksize:
                    # Split large files
                    return IOPartitionPlan(
                        math.ceil(file_size / blocksize),
                        IOPartitionFlavor.SPLIT_FILES,
                    )
                else:
                    # Fuse small files
                    return IOPartitionPlan(
                        max(blocksize // int(file_size), 1),
                        IOPartitionFlavor.FUSED_FILES,
                    )

    # TODO: Use file sizes for csv and json
    return IOPartitionPlan(1, IOPartitionFlavor.SINGLE_FILE)


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
    _n_non_child_args = 13
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
            base_scan.schema,
            base_scan.typ,
            base_scan.reader_options,
            base_scan.paths,
            base_scan.with_columns,
            base_scan.skip_rows,
            base_scan.n_rows,
            base_scan.row_index,
            base_scan.include_file_paths,
            base_scan.predicate,
            base_scan.parquet_options,
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
        *,
        context: IRExecutionContext,
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
            context=context,
        )


@lower_ir_node.register(Empty)
def _(
    ir: Empty, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    return ir, {ir: PartitionInfo(count=1)}  # pragma: no cover


@lower_ir_node.register(Scan)
def _(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    partition_info: MutableMapping[IR, PartitionInfo]
    config_options = rec.state["config_options"]

    # RapidsMPF runtime: Use rapidsmpf-specific lowering
    if (
        config_options.executor.name == "streaming"
        and config_options.executor.runtime == "rapidsmpf"
    ):  # pragma: no cover; Requires rapidsmpf runtime
        from cudf_polars.experimental.rapidsmpf.io import lower_scan_rapidsmpf

        return lower_scan_rapidsmpf(ir, rec)

    if (
        ir.typ in ("csv", "parquet", "ndjson")
        and ir.n_rows == -1
        and ir.skip_rows == 0
        and ir.row_index is None
    ):
        plan = scan_partition_plan(ir, rec.state["stats"], config_options)
        paths = list(ir.paths)
        if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
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

    __slots__ = ("sink", "sink_to_directory")
    _non_child = ("schema", "sink", "sink_to_directory")
    _n_non_child_args = 0

    sink: Sink
    sink_to_directory: bool

    def __init__(
        self,
        schema: Schema,
        sink: Sink,
        sink_to_directory: bool,  # noqa: FBT001
        df: IR,
    ) -> None:
        # Order must match ``_non_child`` + ``children`` so :meth:`Node.__reduce__`
        # / ``reconstruct`` round-trip over pickling (e.g. Dask workers).
        self.schema = schema
        self.sink = sink
        self.sink_to_directory = sink_to_directory
        self._non_child_args = ()
        self.children = (df,)

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        return (type(self), self.sink, *self.children)


@lower_ir_node.register(Sink)
def _(
    ir: Sink, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])
    executor_options = rec.state["config_options"].executor

    assert executor_options.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node'"
    )

    # TODO: Support cloud storage
    if (
        Path(ir.path).exists()
        and executor_options.sink_to_directory
        and executor_options.cluster in (Cluster.SINGLE, Cluster.DISTRIBUTED)
    ):
        # This lowering-time check can't be performed with the new spmd / ray / dask
        # clusters, which lower on each worker independently. There's a race condition
        # between each worker performing this check that the path doesn't yet exist,
        # and the sink operation creating the directory at the start of execution.
        raise NotImplementedError(
            f"Trying to sink to an existing directory: {ir.path}."
            "Writing to an existing path is not supported when sinking "
            "to a directory. If you are using the 'distributed' scheduler, "
            "please remove the target directory before calling 'collect'. "
        )

    sink_to_directory = executor_options.sink_to_directory
    assert sink_to_directory is not None  # set in StreamingExecutor.__post_init__
    new_node = StreamingSink(
        ir.schema,
        ir.reconstruct([child]),
        sink_to_directory,
        child,
    )
    partition_info[new_node] = partition_info[child]
    return new_node, partition_info


def _prepare_sink_directory(path: str) -> None:
    """Prepare for a multi-partition sink."""
    # TODO: Support cloud storage
    Path(path).mkdir(parents=True, exist_ok=True)


def _sink_to_directory(
    schema: Schema,
    kind: str,
    path: str,
    parquet_options: ParquetOptions,
    options: dict[str, Any],
    df: DataFrame,
    ready: None,
    context: IRExecutionContext,
) -> DataFrame:
    """Sink a partition to a new file."""
    return Sink.do_evaluate(
        schema, kind, path, parquet_options, options, df, context=context
    )


def _sink_to_parquet_file(
    path: str,
    options: dict[str, Any],
    writer: plc.io.parquet.ChunkedParquetWriter | None,
    df: DataFrame,
) -> plc.io.parquet.ChunkedParquetWriter:
    """Sink a partition to an open Parquet file."""
    # Set up a new chunked Parquet writer if necessary.
    if writer is None:
        metadata = Sink._make_parquet_metadata(df)
        sink = plc.io.types.SinkInfo([path])
        builder = Sink._apply_parquet_writer_options(
            plc.io.parquet.ChunkedParquetWriterOptions.builder(sink), options
        )
        writer_options = builder.metadata(metadata).build()
        writer = plc.io.parquet.ChunkedParquetWriter.from_options(
            writer_options, stream=df.stream
        )

    # Append to the open Parquet file.
    assert isinstance(writer, plc.io.parquet.ChunkedParquetWriter), (
        "ChunkedParquetWriter is required."
    )
    writer.write(df.table)

    return writer


@overload
def _sink_to_file(
    kind: Literal["Parquet"],
    path: str,
    options: dict[str, Any],
    writer_state: plc.io.parquet.ChunkedParquetWriter,
    df: DataFrame,
) -> plc.io.parquet.ChunkedParquetWriter: ...


@overload
def _sink_to_file(
    kind: str,
    path: str,
    options: dict[str, Any],
    writer_state: None,
    df: DataFrame,
) -> Literal[True]: ...


def _sink_to_file(
    kind: str,
    path: str,
    options: dict[str, Any],
    writer_state: Any,
    df: DataFrame,
) -> Literal[True] | plc.io.parquet.ChunkedParquetWriter:
    """Sink a partition to an open file."""
    if kind == "Parquet":
        # Parquet writer will pass along a
        # ChunkedParquetWriter "writer state".
        return _sink_to_parquet_file(
            path,
            options,
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
            # Path.open returns IO[Any] but SinkInfo needs more specific IO types
            sink = plc.io.types.SinkInfo([f])  # type: ignore[arg-type]
            Sink._write_csv(sink, use_options, df)
    elif kind == "Json" if POLARS_VERSION_LT_137 else "NDJson":
        mode = "wb" if writer_state is None else "ab"
        with Path.open(Path(path), mode) as f:
            # Path.open returns IO[Any] but SinkInfo needs more specific IO types
            sink = plc.io.types.SinkInfo([f])  # type: ignore[arg-type]
            Sink._write_json(sink, df)
    else:  # pragma: no cover; Shouldn't get here.
        raise NotImplementedError(f"{kind} not yet supported in _sink_to_file")

    return True


def _finalize_file_sink(
    kind: str,
    writer_state: Any,
    df: DataFrame,
) -> DataFrame:
    """Finalize the file sink by closing the writer."""
    if kind == "Parquet" and writer_state is not None:
        writer_state.close([])
    return df.slice((0, 0))


def _file_sink_graph(
    ir: StreamingSink,
    partition_info: MutableMapping[IR, PartitionInfo],
    context: IRExecutionContext,
) -> MutableMapping[Any, Any]:
    """Sink to a single file."""
    name = get_key_name(ir)
    count = partition_info[ir].count
    child_name = get_key_name(ir.children[0])
    sink = ir.sink
    if count == 1:
        return {
            (name, 0): (
                partial(sink.do_evaluate, context=context),
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
            None if i == 0 else (sink_name, i - 1),  # Writer state
            (child_name, i),
        )
        for i in range(count)
    }

    # Finalize task closes the writer after all chunks are written
    graph[(sink_name, "finalize")] = (
        _finalize_file_sink,
        sink.kind,
        (sink_name, count - 1),  # Writer state from last task
        (child_name, count - 1),  # Last source df for creating empty result
    )

    # Make sure final tasks point to finalize task
    graph.update({(name, i): (sink_name, "finalize") for i in range(count)})
    return graph


def _directory_sink_graph(
    ir: StreamingSink,
    partition_info: MutableMapping[IR, PartitionInfo],
    context: IRExecutionContext,
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
            sink.parquet_options,
            sink.options,
            (child_name, i),
            setup_name,
            context,
        )
        for i in range(count)
    }
    graph[setup_name] = (_prepare_sink_directory, sink.path)
    return graph


@generate_ir_tasks.register(StreamingSink)
def _(
    ir: StreamingSink,
    partition_info: MutableMapping[IR, PartitionInfo],
    context: IRExecutionContext,
) -> MutableMapping[Any, Any]:
    if ir.sink_to_directory:
        return _directory_sink_graph(ir, partition_info, context=context)
    else:
        return _file_sink_graph(ir, partition_info, context=context)


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
    row_count: int | None
    """Total row-count estimate."""
    num_row_groups_per_file: tuple[int, ...]
    """Number of row groups in each sampled file."""
    mean_size_per_file: dict[str, int]
    """Average column storage size in a single file."""
    column_names: tuple[str, ...]
    """All column names found it the dataset."""
    sample_paths: tuple[str, ...]
    """Sampled file paths."""

    def __init__(self, paths: tuple[str, ...], max_footer_samples: int):
        self.paths = paths
        self.max_footer_samples = max_footer_samples
        self.row_count = None
        self.num_row_groups_per_file = ()
        self.mean_size_per_file = {}
        self.column_names = ()
        stride = (
            max(1, int(len(paths) / max_footer_samples)) if max_footer_samples else 1
        )
        self.sample_paths = paths[: stride * max_footer_samples : stride]

        if not self.sample_paths:
            # No paths to sample from
            # TODO: This requires row_count to be nullable. Why do we allow empty paths?
            return

        total_file_count = len(self.paths)
        sampled_file_count = len(self.sample_paths)
        sample_metadata = plc.io.parquet_metadata.read_parquet_metadata(
            plc.io.SourceInfo(list(self.sample_paths))
        )

        if total_file_count == sampled_file_count:
            row_count = sample_metadata.num_rows()
        else:
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
            name: int(statistics.mean(sizes))
            for name, sizes in column_sizes_per_file.items()
        }
        self.num_row_groups_per_file = tuple(num_row_groups_per_sampled_file)
        self.row_count = row_count


def _sample_rg_sizes(
    metadata: ParquetMetadata,
    target_cols: list[str],
    max_row_group_samples: int,
) -> dict[str, int]:
    """Return mean uncompressed bytes per row-group for each column in target_cols."""
    sample_paths = metadata.sample_paths
    num_row_groups_per_file = metadata.num_row_groups_per_file
    if not sample_paths or len(num_row_groups_per_file) != len(sample_paths):
        return {}  # pragma: no cover

    n_sampled = 0
    samples: defaultdict[str, list[int]] = defaultdict(list)
    for path, num_rgs in zip(sample_paths, num_row_groups_per_file, strict=True):
        for rg_id in range(num_rgs):
            n_sampled += 1
            samples[path].append(rg_id)
            if n_sampled == max_row_group_samples:
                break
        if n_sampled == max_row_group_samples:
            break

    if not n_sampled:
        return {}  # pragma: no cover

    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(list(samples))
    ).build()
    options.set_column_names(target_cols)
    options.set_row_groups(list(samples.values()))
    stream = get_cuda_stream()
    tbl_w_meta = plc.io.parquet.read_parquet(options, stream=stream)
    result = {
        name: column.device_buffer_size() // n_sampled
        for name, column in zip(
            tbl_w_meta.column_names(include_children=False),
            tbl_w_meta.columns,
            strict=True,
        )
    }
    stream.synchronize()
    return result


class ParquetSourceInfo:
    """Parquet datasource information, fully computed at construction time."""

    type: Literal["parquet"] = "parquet"

    def __init__(
        self, row_count: int | None, per_file_means: dict[str, int] | None = None
    ):
        if per_file_means is None:
            per_file_means = {}

        self.row_count = row_count
        self.per_file_means = per_file_means

    @classmethod
    def from_paths(
        cls,
        paths: tuple[str, ...],
        needed_cols: frozenset[str],
        max_footer_samples: int,
        max_row_group_samples: int,
    ) -> ParquetSourceInfo:
        """Build a ParquetSourceInfo from a list of paths."""
        metadata = ParquetMetadata(paths, max_footer_samples)
        row_count = metadata.row_count

        file_count = len(paths)
        per_file_means: dict[str, int] = {}

        if not (file_count and row_count and needed_cols):
            return cls(row_count, {})

        # Floor on size: dictionary encoding can make in-memory size much larger
        # than what the compressed footer metadata reports.
        min_floor = max(1, row_count // file_count)
        suspicious: list[str] = []

        for col in needed_cols:
            footer_mean = metadata.mean_size_per_file.get(col)
            if footer_mean is None:
                continue
            if footer_mean < min_floor:
                suspicious.append(col)
            else:
                per_file_means[col] = footer_mean

        if suspicious and max_row_group_samples > 0:
            rg_sizes = _sample_rg_sizes(metadata, suspicious, max_row_group_samples)
            mean_rg_count = (
                statistics.mean(metadata.num_row_groups_per_file)
                if metadata.num_row_groups_per_file
                else 1
            )
            for col in suspicious:
                rg_size = rg_sizes.get(col)
                per_file_means[col] = (
                    max(min_floor, int(rg_size * mean_rg_count))
                    if rg_size
                    else min_floor
                )
        else:
            for col in suspicious:
                per_file_means[col] = min_floor

        return cls(row_count, per_file_means)

    def column_storage_size(self, column: str) -> int | None:
        """Return the average storage size for a single column in one file."""
        return self.per_file_means.get(column)

    def serialize(self) -> SerializedDataSourceInfo:
        """Return JSON-serializable representation of the data source info."""
        return {
            "type": self.type,
            "row_count": self.row_count,
            "per_file_means": self.per_file_means,
        }

    @classmethod
    def deserialize(cls, data: SerializedDataSourceInfo) -> ParquetSourceInfo:
        """Deserialize a ParquetSourceInfo from a dictionary."""
        if data["type"] != "parquet":
            raise ValueError(f"Expected ParquetSourceInfo, got {data['type']}")
        return cls(data["row_count"], data["per_file_means"])


class DataFrameSourceInfo:
    """
    In-memory DataFrame source information.

    Parameters
    ----------
    row_count
        Exact row-count for the polars dataframe.
    """

    type: Literal["dataframe"] = "dataframe"

    def __init__(self, row_count: int):
        self.row_count = row_count

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> DataFrameSourceInfo:
        """Build a DataFrameSourceInfo from a polars dataframe."""
        return cls(df.height)

    def column_storage_size(self, column: str) -> int | None:
        """Return the average storage size for a single column in one file."""
        return None

    def serialize(self) -> SerializedDataSourceInfo:
        """Return JSON-serializable representation of the data source info."""
        return {
            "type": self.type,
            "row_count": self.row_count,
            "per_file_means": None,
        }

    @classmethod
    def deserialize(cls, data: SerializedDataSourceInfo) -> DataFrameSourceInfo:
        """Deserialize a DataFrameSourceInfo from a dictionary."""
        if data["type"] != "dataframe":
            raise ValueError(f"Expected DataFrameSourceInfo, got {data['type']}")
        if data["row_count"] is None:
            raise ValueError("Row count is required for DataFrameSourceInfo")
        return cls(data["row_count"])


@functools.cache
def _build_parquet_source(
    paths: tuple[str, ...],
    needed_cols: frozenset[str],
    max_footer_samples: int,
    max_row_group_samples: int,
) -> ParquetSourceInfo:
    """Return cached, fully-computed Parquet datasource information."""
    return ParquetSourceInfo.from_paths(
        paths, needed_cols, max_footer_samples, max_row_group_samples
    )


def _build_source_info(
    ir: Scan | DataFrameScan,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    needed_cols: frozenset[str] | None = None,
) -> DataSourceInfo:
    """Return DataSourceInfo for a Scan or DataFrameScan node."""
    if isinstance(ir, DataFrameScan):
        return DataFrameSourceInfo.from_polars(pl.DataFrame._from_pydf(ir.df))
    elif isinstance(ir, Scan) and ir.typ == "parquet":
        max_footer = config_options.parquet_options.max_footer_samples
        max_rg = config_options.parquet_options.max_row_group_samples
        needed_cols = frozenset(ir.schema) if needed_cols is None else needed_cols
        paths = tuple(ir.paths)
        return _build_parquet_source(paths, needed_cols, max_footer, max_rg)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported Scan type: {ir.typ}")


def _clear_source_info_cache() -> None:
    """Clear DataSourceInfo caches."""
    # TODO: Avoid clearing the cache if we can
    # check that the underlying data hasn't changed.
    _build_parquet_source.cache_clear()
