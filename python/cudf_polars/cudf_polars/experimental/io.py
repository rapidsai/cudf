# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition IO Logic."""

from __future__ import annotations

import dataclasses
import enum
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
    ColumnSourceStats,
    PartitionInfo,
    StatsCollector,
    get_key_name,
)
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.experimental.base import ColumnStats
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions, ParquetOptions


# Cache source stats for each tuple of path names
_SOURCE_STATS_CACHE: dict[tuple[str, ...], dict[str, ColumnSourceStats]] = {}
_SOURCE_STATS_CACHE_KEYS: list[tuple[str, ...]] = []
_SOURCE_STATS_CACHE_MAX_ITEMS: int = 10


def _update_source_stats_cache(
    key: tuple[str, ...],
    value: dict[str, ColumnSourceStats],
) -> None:
    """Update _SOURCE_STATS_CACHE with LRU eviction."""
    if key in _SOURCE_STATS_CACHE_KEYS:
        _SOURCE_STATS_CACHE_KEYS.remove(key)

    if key not in _SOURCE_STATS_CACHE and (
        len(_SOURCE_STATS_CACHE) >= _SOURCE_STATS_CACHE_MAX_ITEMS
    ):
        del _SOURCE_STATS_CACHE[_SOURCE_STATS_CACHE_KEYS.pop(0)]

    _SOURCE_STATS_CACHE_KEYS.append(key)
    _SOURCE_STATS_CACHE[key] = value


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
    def from_scan(
        ir: Scan,
        *,
        config_options: ConfigOptions,
        column_stats: dict[str, ColumnStats],
    ) -> ScanPartitionPlan:
        """Extract the partitioning plan of a Scan operation."""
        if ir.typ == "parquet":
            assert config_options.executor.name == "streaming", (
                "'in-memory' executor not supported in 'ScanPartitionPlan'"
            )

            blocksize: int = config_options.executor.target_partition_size
            column_sizes = []
            for name, cs in column_stats.items():
                if (
                    name in ir.schema
                    and cs.source_stats is not None
                    and cs.source_stats.storage_size_per_file is not None
                ):
                    column_sizes.append(cs.source_stats.storage_size_per_file)

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


def _sample_pq_stats(
    ir: Scan, config_options: ConfigOptions
) -> dict[str, ColumnSourceStats]:
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in '_sample_pq_stats"
    )

    max_file_samples = config_options.executor.parquet_metadata_samples
    max_rg_samples = config_options.executor.parquet_rowgroup_samples

    total_file_count = len(ir.paths)
    stride = max(1, int(total_file_count / max_file_samples))
    sample_paths = ir.paths[: stride * max_file_samples : stride]
    sampled_file_count = len(sample_paths)
    exact_stats: tuple[str, ...] = ()

    # Check table-stats cache
    source_stats_cached: MutableMapping[str, ColumnSourceStats]
    try:
        source_stats_cached = _SOURCE_STATS_CACHE[tuple(ir.paths)]
    except KeyError:
        source_stats_cached = {}
    finally:
        source_stats = source_stats_cached

    if need_columns := (set(ir.schema) - source_stats_cached.keys()):
        # Still need columns missing from the cache
        sample_metadata = plc.io.parquet_metadata.read_parquet_metadata(
            plc.io.SourceInfo(sample_paths)
        )

        if total_file_count == sampled_file_count:
            # We know the "exact" cardinality from our sample
            cardinality = sample_metadata.num_rows()
            exact_stats = ("cardinality",)
        else:
            # We must estimate/extrapolate the cardinality from our sample
            num_rows_per_sampled_file = int(
                sample_metadata.num_rows() / sampled_file_count
            )
            cardinality = num_rows_per_sampled_file * total_file_count

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

        # Revise need_columns, since some columns may not be in the file
        need_columns = need_columns.intersection(column_sizes_per_file)

    if need_columns:
        # We have un-cached column metadata to process

        # Calculate the `mean_uncompressed_size_per_file` for each column
        mean_uncompressed_size_per_file = {
            name: statistics.mean(sizes)
            for name, sizes in column_sizes_per_file.items()
        }

        # Collect real unique-count of first row-group
        unique_count_estimates: dict[str, int] = {}
        unique_fraction_estimates: dict[str, float] = {}
        if max_rg_samples > 0:
            n = 0
            samples: defaultdict[str, list[int]] = defaultdict(list)
            for path, num_rgs in zip(
                sample_paths, num_row_groups_per_sampled_file, strict=True
            ):
                for rg_id in range(num_rgs):
                    n += 1
                    samples[path].append(rg_id)
                    if n == max_rg_samples:
                        break
                if n == max_rg_samples:
                    break
            options = plc.io.parquet.ParquetReaderOptions.builder(
                plc.io.SourceInfo(list(samples))
            ).build()
            options.set_columns([c for c in ir.schema if c in need_columns])
            options.set_row_groups(list(samples.values()))
            tbl_w_meta = plc.io.parquet.read_parquet(options)
            row_group_num_rows = tbl_w_meta.tbl.num_rows()
            for name, column in zip(
                tbl_w_meta.column_names(), tbl_w_meta.columns, strict=True
            ):
                if name in need_columns:
                    row_group_unique_count = plc.stream_compaction.distinct_count(
                        column,
                        plc.types.NullPolicy.INCLUDE,
                        plc.types.NanPolicy.NAN_IS_NULL,
                    )
                    unique_fraction_estimates[name] = max(
                        min(1.0, row_group_unique_count / row_group_num_rows),
                        0.00001,
                    )
                    # Assume that if every row is unique then this is a
                    # primary key otherwise it's a foreign key and we
                    # can't use the single row group count estimate
                    # Example, consider a "foreign" key that has 100
                    # unique values. If we sample from a single row group,
                    # we likely obtain a unique count of 100. But we can't
                    # necessarily deduce that that means that the unique
                    # count is 100 / num_rows_in_group * num_rows_in_file
                    if row_group_unique_count == row_group_num_rows:
                        unique_count_estimates[name] = cardinality

        # Check that the cached stats have the same row-count estimate
        if source_stats_cached:
            assert (
                cardinality == next(iter(source_stats_cached.values())).cardinality
            ), "Unexpected cardinality in cache."

        # Construct estimated column statistics
        source_stats = {
            name: ColumnSourceStats(
                cardinality=cardinality,
                storage_size_per_file=mean_uncompressed_size_per_file[name],
                unique_count=unique_count_estimates.get(name),
                unique_fraction=unique_fraction_estimates.get(name),
                exact=exact_stats,
            )
            for name in need_columns
        }

        if source_stats_cached:
            # Combine new and cached column stats
            source_stats = source_stats_cached | source_stats

    if source_stats:
        # Update the cache
        _update_source_stats_cache(tuple(ir.paths), source_stats)

    # Return relevant source stats
    return {name: css for name, css in source_stats.items() if name in ir.schema}


@lower_ir_node.register(Scan)
def _(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    partition_info: MutableMapping[IR, PartitionInfo]
    config_options = rec.state["config_options"]
    if ir.typ in ("csv", "parquet", "ndjson") and ir.n_rows == -1 and ir.skip_rows == 0:
        stats_collector = rec.state.get("stats")
        assert isinstance(stats_collector, StatsCollector), (
            f"Expected StatsCollector, got {type(stats_collector)}"
        )
        plan = ScanPartitionPlan.from_scan(
            ir,
            config_options=config_options,
            column_stats=stats_collector.column_stats.get(ir, {}),
        )
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
