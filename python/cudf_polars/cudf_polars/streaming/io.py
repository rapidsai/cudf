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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.dsl.ir import (
    IR,
    DataFrameScan,
    Empty,
    Scan,
    Sink,
    _prepare_parquet_predicate,
)
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars
from cudf_polars.streaming.base import (
    IOPartitionFlavor,
    IOPartitionPlan,
    PartitionInfo,
    SerializedDataSourceInfo,
)
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.utils import _dynamic_planning_on
from cudf_polars.utils.config import Cluster
from cudf_polars.utils.cuda_stream import get_cuda_stream
from cudf_polars.utils.versions import POLARS_VERSION_LT_137

if TYPE_CHECKING:
    from collections.abc import Hashable, MutableMapping

    import pylibcudf.expressions as plc_expr
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.base import (
        DataSourceInfo,
        SerializedDataSourceInfo,
        StatsCollector,
    )
    from cudf_polars.streaming.dispatch import LowerIRTransformer
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

    # NOTE: We calculate the expected partition count
    # to help trigger fallback warnings in lower_ir_graph.
    # The generate_ir_sub_network logic is NOT required
    # to obey this partition count. However, the count
    # WILL match after an IO operation (for now).
    rows_per_partition = config_options.executor.max_rows_per_partition
    nrows = max(ir.df.shape()[0], 1)
    count = math.ceil(nrows / rows_per_partition)

    return ir, {ir: PartitionInfo(count=count)}


def scan_partition_plan(
    ir: Scan, stats: StatsCollector, config_options: ConfigOptions[StreamingExecutor]
) -> IOPartitionPlan:
    """Extract the partitioning plan of a Scan operation."""
    if ir.typ == "parquet":
        blocksize: int = config_options.executor.target_partition_size
        single_file = len(ir.paths) == 1
        if source := stats.scan_stats.get(ir):
            column_sizes = [
                sz
                for col in ir.schema
                if (sz := source.column_storage_size(col)) is not None
            ]
            if (file_size := sum(column_sizes)) > 0:
                if file_size > blocksize or single_file:
                    # A single file always uses SplitScan even if it is smaller than
                    # the blocksize, so that hybrid scan can be used on it.
                    factor = math.ceil(file_size / blocksize)
                    return IOPartitionPlan(
                        factor,
                        IOPartitionFlavor.SPLIT_FILES,
                        estimated_chunk_bytes=file_size // factor,
                    )
                else:
                    # Fuse small files
                    factor = max(blocksize // int(file_size), 1)
                    return IOPartitionPlan(
                        factor,
                        IOPartitionFlavor.FUSED_FILES,
                        estimated_chunk_bytes=file_size * factor,
                    )

        if single_file:
            return IOPartitionPlan(1, IOPartitionFlavor.SPLIT_FILES)

    # TODO: Use file sizes for csv and json
    return IOPartitionPlan(1, IOPartitionFlavor.SINGLE_FILE)


def expand_scan_for_rank(
    ir: Scan,
    plan: IOPartitionPlan,
    *,
    rank: int,
    nranks: int,
    parquet_options: ParquetOptions,
) -> list[SplitScan | FusedScan]:
    """
    Expand a Scan node into rank-local SplitScan and FusedScan operations.

    Parameters
    ----------
    ir
        The Scan node to expand.
    plan
        The IO partitioning plan for the scan.
    rank
        Rank of the current worker.
    nranks
        Number of workers.
    parquet_options
        Parquet reader options.

    Returns
    -------
    list[SplitScan | FusedScan]
        Rank-local scan operations.
    """
    scans: list[SplitScan | FusedScan] = []
    if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
        count = plan.factor * len(ir.paths)
        local_count = math.ceil(count / nranks)
        local_offset = local_count * rank
        path_offset = local_offset // plan.factor
        path_end = math.ceil((local_offset + local_count) / plan.factor)
        path_count = path_end - path_offset
        local_paths = ir.paths[path_offset : path_offset + path_count]
        sindex = local_offset % plan.factor
        splits_created = 0
        for path in local_paths:
            while sindex < plan.factor and splits_created < local_count:
                scans.append(
                    SplitScan(
                        ir.schema,
                        ir,
                        [path],
                        sindex,
                        plan.factor,
                        parquet_options,
                    )
                )
                sindex += 1
                splits_created += 1
            sindex = 0

    else:
        count = math.ceil(len(ir.paths) / plan.factor)
        local_count = math.ceil(count / nranks)
        local_offset = local_count * rank
        paths_offset_start = local_offset * plan.factor
        paths_offset_end = paths_offset_start + plan.factor * local_count
        for offset in range(paths_offset_start, paths_offset_end, plan.factor):
            local_paths = ir.paths[offset : offset + plan.factor]
            if len(local_paths) > 0:
                scans.append(FusedScan(ir.schema, ir, local_paths, parquet_options))

    return scans


# Per-file cache: row-group row counts, keyed by file path.
# Avoids a redundant read_parquet_metadata call for every split of the same
# file in SplitScan.do_evaluate; populated on first encounter.
_row_group_num_rows_cache: dict[str, list[int]] = {}

# Per-file cache: plain FileMetaData (footer only, no page index).
# Avoids a redundant read_parquet_footers call for every split of the same
# file; populated on first encounter.
_parquet_footer_cache: dict[str, Any] = {}

# TODO: Once footer prefetch (#22700) lands, we'll get
# the metadata from IRExecutionContext footer cache
_hybrid_scan_metadata_cache: dict[str, Any] = {}


def _fetch_byte_ranges(
    paths: list[str],
    byte_ranges: list[plc.io.text.ByteRangeInfo],
    stream: Stream,
) -> list[plc.gpumemoryview]:
    # TODO: Accept a pinned-host Datasource pre-fetched by the caller so the
    # storage I/O overlaps with GPU work for better pipelining.
    return plc.io.parquet_io_utils.fetch_byte_ranges_to_device(
        plc.io.SourceInfo(paths), byte_ranges, stream=stream
    )


def _read_with_hybrid_scan(
    schema: Schema,
    paths: list[str],
    with_columns: list[str] | None,
    plc_filter: plc_expr.Expression,
    row_group_indices: list[int],
    stream: Stream,
    file_metadata: plc.io.parquet_metadata.FileMetaData,
    *,
    split_index: int = 0,
    total_splits: int = 1,
    stats_pruning: bool = True,
) -> DataFrame:
    """Two-pass parquet read via HybridScanReader for a row-group-aligned split."""
    assert plc_filter is not None
    assert len(paths) == 1, (
        "hybrid scan only supported for SplitScan; one physical file"
    )
    with nvtx_annotate_cudf_polars(
        message=f"HybridScan: {paths[0]} [{split_index + 1}/{total_splits}]"
    ):
        options = (
            plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo(paths))
            .decimal_width(plc.TypeId.DECIMAL128)
            .build()
        )
        if with_columns is not None:
            options.set_column_names(with_columns)
        options.set_filter(plc_filter)

        # Parse the file metadata once per file and share it across the file's
        # splits, rather than re-parsing and copying it per split.
        metadata = _hybrid_scan_metadata_cache.get(paths[0])
        if metadata is None:
            metadata = plc.io.experimental.HybridScanMetadata.from_parquet_metadata(
                file_metadata, options
            )
            _hybrid_scan_metadata_cache[paths[0]] = metadata
        reader = plc.io.experimental.HybridScanReader.from_metadata(metadata)

        if stats_pruning:
            row_group_indices = reader.filter_row_groups_with_stats(
                row_group_indices, options, stream=stream
            )

            if row_group_indices:
                bloom_ranges, _ = reader.secondary_filters_byte_ranges(
                    row_group_indices, options
                )
                if bloom_ranges:
                    bloom_chunks = _fetch_byte_ranges(paths, bloom_ranges, stream)
                    row_group_indices = reader.filter_row_groups_with_bloom_filters(
                        bloom_chunks, row_group_indices, options, stream=stream
                    )

        if not row_group_indices:
            col_names = with_columns if with_columns is not None else list(schema)
            return DataFrame(
                [
                    Column(
                        plc.column_factories.make_empty_column(
                            schema[name].plc_type, stream=stream
                        ),
                        dtype=schema[name],
                        name=name,
                    )
                    for name in col_names
                ],
                stream=stream,
            )

        # TODO: Consider implementing page-index stats pruning. For SplitScans, we can
        # reuse the same page index for all splits of the same file, so the overhead of
        # reading the page index can be amortized. For FusedScans, we would need to read
        # the page index for all files, which may be too expensive.
        row_mask = reader.build_all_true_row_mask(row_group_indices, stream=stream)

        filter_ranges = reader.filter_column_chunks_byte_ranges(
            row_group_indices, options
        )
        filter_chunks = _fetch_byte_ranges(paths, filter_ranges, stream)
        filter_tbl_w_meta = reader.materialize_filter_columns(
            row_group_indices,
            filter_chunks,
            row_mask,
            plc.io.experimental.UseDataPageMask.YES,
            options,
            stream=stream,
        )

        payload_ranges = reader.payload_column_chunks_byte_ranges(
            row_group_indices, options
        )
        payload_chunks = _fetch_byte_ranges(paths, payload_ranges, stream)
        payload_tbl_w_meta = reader.materialize_payload_columns(
            row_group_indices,
            payload_chunks,
            row_mask,
            plc.io.experimental.UseDataPageMask.YES,
            options,
            stream=stream,
        )

        filter_names = filter_tbl_w_meta.column_names(include_children=False)
        payload_names = payload_tbl_w_meta.column_names(include_children=False)
        filter_df = DataFrame.from_table(
            filter_tbl_w_meta.tbl,
            filter_names,
            [schema[n] for n in filter_names],
            stream=stream,
        )
        payload_df = DataFrame.from_table(
            payload_tbl_w_meta.tbl,
            payload_names,
            [schema[n] for n in payload_names],
            stream=stream,
        )
        # Ensure the decode kernels are finished before
        # filter_chunks and payload_chunks go out of scope
        stream.synchronize()
        return DataFrame(
            [*filter_df.columns, *payload_df.columns], stream=stream
        ).select(list(schema.keys()))


class SplitScan(IR):
    """
    Input from a split file.

    This class wraps a single-file ``Scan`` object. At
    IO/evaluation time, this class will only perform
    a partial read of the underlying file. The range
    (skip_rows and n_rows) is calculated at IO time.
    """

    __slots__ = (
        "base_scan",
        "parquet_options",
        "paths",
        "schema",
        "split_index",
        "total_splits",
    )
    _non_child = (
        "schema",
        "base_scan",
        "paths",
        "split_index",
        "total_splits",
        "parquet_options",
    )
    _n_non_child_args = 13
    base_scan: Scan
    """Scan operation this node is based on."""
    paths: list[str]
    """File path for this split task."""
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
        paths: list[str],
        split_index: int,
        total_splits: int,
        parquet_options: ParquetOptions,
    ):
        self.schema = schema
        self.base_scan = base_scan
        self.paths = paths
        self.split_index = split_index
        self.total_splits = total_splits
        self._non_child_args = (
            split_index,
            total_splits,
            base_scan.schema,
            base_scan.typ,
            base_scan.reader_options,
            paths,
            base_scan.with_columns,
            base_scan.skip_rows,
            base_scan.n_rows,
            base_scan.row_index,
            base_scan.include_file_paths,
            base_scan.predicate,
            parquet_options,
        )
        self.parquet_options = parquet_options
        self.children = ()
        if base_scan.typ not in ("parquet",):  # pragma: no cover
            raise NotImplementedError(
                f"Unhandled Scan type for file splitting: {base_scan.typ}"
            )

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        return (
            type(self),
            tuple(self.schema.items()),
            self.base_scan.get_hashable(),
            tuple(self.paths),
            self.split_index,
            self.total_splits,
            self.parquet_options,
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

        row_group_num_rows = _row_group_num_rows_cache.get(paths[0])
        if row_group_num_rows is None:
            row_group_num_rows = [
                rg["num_rows"]
                for rg in plc.io.parquet_metadata.read_parquet_metadata(
                    plc.io.SourceInfo(paths)
                ).rowgroup_metadata()
            ]
            _row_group_num_rows_cache[paths[0]] = row_group_num_rows

        total_row_groups = len(row_group_num_rows)
        if total_splits <= total_row_groups:
            # We have enough row-groups in the file to align
            # all "total_splits" of our reads with row-group
            # boundaries. Calculate which row-groups to include
            # in the current read, and use metadata to translate
            # the row-group indices to "skip_rows" and "n_rows".
            rg_stride = total_row_groups // total_splits
            skip_rgs = rg_stride * split_index
            skip_rows = sum(row_group_num_rows[:skip_rgs])
            n_rows = sum(row_group_num_rows[skip_rgs : skip_rgs + rg_stride])
            # TODO: Investigate re-enabling for some of these
            # paths. Needs performance investigation.
            if (
                parquet_options.use_hybrid_scan
                and row_index is None
                and include_file_paths is None
                and predicate is not None
            ):
                stream = context.get_cuda_stream()
                plc_filter = to_parquet_filter(
                    _prepare_parquet_predicate(
                        predicate.value, paths, schema, with_columns
                    ),
                    stream=stream,
                )
                if plc_filter is not None:
                    end_rg = (
                        total_row_groups
                        if split_index == total_splits - 1
                        else skip_rgs + rg_stride
                    )
                    # Reuse the cached plain footer, reading from disk only on
                    # the first split that encounters this file.
                    file_metadata = _parquet_footer_cache.get(paths[0])
                    if file_metadata is None:
                        [file_metadata] = plc.io.parquet_metadata.read_parquet_footers(
                            plc.io.SourceInfo(paths)
                        )
                        _parquet_footer_cache[paths[0]] = file_metadata
                    return _read_with_hybrid_scan(
                        schema,
                        paths,
                        with_columns,
                        plc_filter,
                        list(range(skip_rgs, end_rg)),
                        stream,
                        file_metadata,
                        split_index=split_index,
                        total_splits=total_splits,
                        stats_pruning=parquet_options.hybrid_scan_stats_pruning,
                    )

        else:
            # There are not enough row-groups to align
            # all "total_splits" of our reads with row-group
            # boundaries. Use metadata to directly calculate
            # "skip_rows" and "n_rows" for the current read.
            total_rows = sum(row_group_num_rows)
            n_rows = total_rows // total_splits
            skip_rows = n_rows * split_index

        # Last split should always read to end of file
        if split_index == (total_splits - 1):
            n_rows = -1

        # Perform the partial read
        with nvtx_annotate_cudf_polars(
            message=f"SplitScan: {paths[0]} [{split_index + 1}/{total_splits}]"
        ):
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


class FusedScan(IR):
    """
    Input from one or more complete files read as a single task.

    Covers both FUSED_FILES (N > 1 small files grouped together) and
    SINGLE_FILE (N = 1).
    """

    __slots__ = (
        "base_scan",
        "parquet_options",
        "paths",
        "schema",
    )
    _non_child = (
        "schema",
        "base_scan",
        "paths",
        "parquet_options",
    )
    _n_non_child_args = 11
    base_scan: Scan
    """Scan operation this node is based on."""
    paths: list[str]
    """File paths assigned to this task."""
    parquet_options: ParquetOptions
    """Parquet-specific options."""

    def __init__(
        self,
        schema: Schema,
        base_scan: Scan,
        paths: list[str],
        parquet_options: ParquetOptions,
    ):
        self.schema = schema
        self.base_scan = base_scan
        self.paths = paths
        self.parquet_options = parquet_options
        self._non_child_args = (
            base_scan.schema,
            base_scan.typ,
            base_scan.reader_options,
            paths,
            base_scan.with_columns,
            base_scan.skip_rows,
            base_scan.n_rows,
            base_scan.row_index,
            base_scan.include_file_paths,
            base_scan.predicate,
            parquet_options,
        )
        self.children = ()

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        return (
            type(self),
            tuple(self.schema.items()),
            self.base_scan.get_hashable(),
            tuple(self.paths),
            self.parquet_options,
        )

    @classmethod
    def do_evaluate(
        cls,
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
        with nvtx_annotate_cudf_polars(message=f"FusedScan: {', '.join(paths)}"):
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


def can_use_native_parquet_node(
    ir: Scan,
    *,
    plan: IOPartitionPlan,
    count: int,
    nranks: int,
    parquet_options: ParquetOptions,
    config_options: ConfigOptions[StreamingExecutor],
) -> bool:
    """
    Determine whether we should use rapidsmpf's native parquet node.

    Parameters
    ----------
    ir
        The Scan node that might need to fall back.
    plan
        The IO partitioning plan.
    count
        The number of partitions associated with this Scan node.
    nranks
        The number of ranks.
    parquet_options
        The parquet options.
    config_options
        The configuration options.

    Returns
    -------
    bool
        Whether to use rapidsmpf's native parquet node.

    Notes
    -----
    Native parquet node is used under the following conditions:

    - Our plan indicates we should split the file into multiple partitions
    - We have more than one rank
    - There's more than one partition or dynamic planning is enabled
    - The file type is parquet
    - The row index is not set
    - File paths are not included
    - The number of rows is not set
    - The skip rows is not set
    """
    distributed_split_files = (
        plan.flavor == IOPartitionFlavor.SPLIT_FILES and nranks > 1
    )

    return (
        parquet_options.use_rapidsmpf_native
        and (count > 1 or _dynamic_planning_on(config_options))
        and ir.typ == "parquet"
        and ir.row_index is None
        and ir.include_file_paths is None
        and ir.n_rows == -1
        and ir.skip_rows == 0
        and not distributed_split_files
    )


@lower_ir_node.register(Scan)
def _(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[StreamingScan, MutableMapping[IR, PartitionInfo]]:
    config_options = rec.state["config_options"]
    parquet_options = config_options.parquet_options
    if (
        ir.typ in ("csv", "parquet", "ndjson")
        and ir.n_rows == -1
        and ir.skip_rows == 0
        and ir.row_index is None
    ):
        # NOTE: We calculate the expected partition count
        # to help trigger fallback warnings in lower_ir_graph.
        # The generate_ir_sub_network logic is NOT required
        # to obey this partition count. However, the count
        # WILL match after an IO operation (for now).
        plan = scan_partition_plan(ir, rec.state["stats"], config_options)
        paths = list(ir.paths)
        if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
            count = plan.factor * len(paths)
        else:
            count = math.ceil(len(paths) / plan.factor)
    else:
        plan = IOPartitionPlan(
            flavor=IOPartitionFlavor.SINGLE_READ, factor=len(ir.paths)
        )
        count = 1

    if not can_use_native_parquet_node(
        ir,
        plan=plan,
        count=count,
        nranks=rec.state["nranks"],
        parquet_options=parquet_options,
        config_options=config_options,
    ):
        parquet_options = dataclasses.replace(parquet_options, chunked=False)

    scans = expand_scan_for_rank(
        ir,
        plan,
        rank=rec.state["rank"],
        nranks=rec.state["nranks"],
        parquet_options=parquet_options,
    )
    new_ir = StreamingScan(scans, ir)
    return new_ir, {new_ir: PartitionInfo(count=count, io_plan=plan)}


class StreamingScan(IR):
    """A streaming scan node."""

    __slots__ = (
        "base_scan",
        "scans",
        "schema",
    )
    _non_child = (
        "scans",
        "base_scan",
    )
    _n_non_child_args = 2
    scans: list[SplitScan | FusedScan]
    base_scan: Scan

    def __init__(self, scans: list[SplitScan | FusedScan], base_scan: Scan):
        self.scans = scans
        self.base_scan = base_scan
        self.schema = base_scan.schema
        self._non_child_args = (scans, base_scan)
        self.children = ()

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        # We don't need to include base_scan / schema, since it's in all the scan nodes.
        return (type(self), *tuple(x.get_hashable() for x in self.scans))

    @classmethod
    def do_evaluate(
        cls,
        scans: list[SplitScan | FusedScan],
        base_scan: Scan,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Raises NotImplementedError for StreamingScan nodes."""
        raise NotImplementedError(
            "StreamingScan.do_evaluate should not be called directly. Call Scan.do_evaluate on each scan node instead."
        )


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
        and executor_options.cluster == Cluster.DEFAULT_SINGLETON
    ):
        # This lowering-time check can't be performed with the spmd / ray / dask
        # clusters, which lower on each worker independently. There's a race condition
        # between each worker performing this check that the path doesn't yet exist,
        # and the sink operation creating the directory at the start of execution.
        raise NotImplementedError(
            f"Trying to sink to an existing directory: {ir.path}. "
            "Writing to an existing path is not supported when sinking "
            "to a directory. Please remove the target directory before "
            "calling 'collect'."
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
    elif kind == ("Json" if POLARS_VERSION_LT_137 else "NDJson"):
        mode = "wb" if writer_state is None else "ab"
        with Path.open(Path(path), mode) as f:
            # Path.open returns IO[Any] but SinkInfo needs more specific IO types
            sink = plc.io.types.SinkInfo([f])  # type: ignore[arg-type]
            Sink._write_json(sink, df)
    else:  # pragma: no cover; Shouldn't get here.
        raise NotImplementedError(f"{kind} not yet supported in _sink_to_file")

    return True


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

    @nvtx_annotate_cudf_polars(message="ParquetMetadata")
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


@nvtx_annotate_cudf_polars(message="_sample_rg_sizes")
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
