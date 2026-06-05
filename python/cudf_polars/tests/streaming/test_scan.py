# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl.ir import (
    Empty,
    IRExecutionContext,
    Scan,
    prefetch_parquet_file_metadata_for_ir,
)
from cudf_polars.dsl.traversal import traversal
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.base import IOPartitionFlavor, IOPartitionPlan
from cudf_polars.streaming.io import SplitScan, StreamingScan, expand_scan_for_rank
from cudf_polars.streaming.parallel import lower_ir_graph
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.config import ConfigOptions, ParquetOptions

if TYPE_CHECKING:
    import concurrent.futures
    from pathlib import Path


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "x": range(3_000),
            "y": ["cat", "dog", "fish"] * 1_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 600,
        }
    )


@pytest.mark.parametrize(
    "fmt, scan_fn",
    [
        ("csv", pl.scan_csv),
        ("ndjson", pl.scan_ndjson),
        ("parquet", pl.scan_parquet),
    ],
)
@pytest.mark.timeout(90)
def test_parallel_scan(tmp_path, df, fmt, scan_fn, streaming_engine):
    make_partitioned_source(df, tmp_path, fmt, n_files=3)
    q = scan_fn(tmp_path)
    assert_gpu_result_equal(q, engine=streaming_engine)


def test_scan_parquet_use_rapidsmpf_native(tmp_path, df, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            target_partition_size=1_000,
            parquet_options={"use_rapidsmpf_native": True},
        ),
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    assert_gpu_result_equal(pl.scan_parquet(tmp_path), engine=streaming_engine)


def test_scan_parquet_prefetch_file_metadata(tmp_path, df, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            target_partition_size=1_000,
            parquet_options={"prefetch_file_metadata": True},
        ),
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=2)
    assert_gpu_result_equal(pl.scan_parquet(tmp_path), engine=streaming_engine)


def test_prefetch_parquet_file_metadata_streaming_scan_children(
    tmp_path,
    df,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    # Small files fused into Scan children (not SplitScan) inside StreamingScan.
    make_partitioned_source(df, tmp_path, "parquet", n_files=3)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": 1_000_000},
        parquet_options={"prefetch_file_metadata": True},
    )
    q = pl.scan_parquet(tmp_path)
    qir = Translator(q._ldf.visit(), engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(engine)
    stats = collect_statistics(qir, config_options, parquet_stats_executor)
    ir, partition_info = lower_ir_graph(
        qir,
        config_options,
        stats,
        rank=0,
        nranks=1,
    )

    streaming_scan = next(
        node for node in traversal([ir]) if isinstance(node, StreamingScan)
    )
    parquet_scans = [
        scan
        for scan in streaming_scan.scans
        if isinstance(scan, Scan) and scan.typ == "parquet"
    ]
    assert parquet_scans
    io_plan = partition_info[streaming_scan].io_plan
    assert io_plan is not None
    assert io_plan.flavor != IOPartitionFlavor.SPLIT_FILES

    context = IRExecutionContext()
    prefetch_parquet_file_metadata_for_ir(ir, context)

    for scan in parquet_scans:
        assert tuple(scan.paths) in context.parquet_file_metadata
        assert context.parquet_file_metadata[tuple(scan.paths)]


def test_prefetch_parquet_file_metadata_streaming_split_scan_children(
    tmp_path,
    df,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    # Large file split into SplitScan children inside StreamingScan.
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": 1_000},
        parquet_options={"prefetch_file_metadata": True},
    )
    q = pl.scan_parquet(tmp_path)
    qir = Translator(q._ldf.visit(), engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(engine)
    stats = collect_statistics(qir, config_options, parquet_stats_executor)
    ir, partition_info = lower_ir_graph(
        qir,
        config_options,
        stats,
        rank=0,
        nranks=1,
    )

    streaming_scan = next(
        node for node in traversal([ir]) if isinstance(node, StreamingScan)
    )
    split_scans = [
        scan
        for scan in streaming_scan.scans
        if isinstance(scan, SplitScan) and scan.base_scan.typ == "parquet"
    ]
    assert split_scans
    io_plan = partition_info[streaming_scan].io_plan
    assert io_plan is not None
    assert io_plan.flavor == IOPartitionFlavor.SPLIT_FILES

    context = IRExecutionContext()
    prefetch_parquet_file_metadata_for_ir(ir, context)

    for scan in split_scans:
        paths = tuple(scan.base_scan.paths)
        assert paths in context.parquet_file_metadata
        assert context.parquet_file_metadata[paths]


def test_prefetch_parquet_file_metadata_split_scan_root(tmp_path, df) -> None:
    # SplitScan is stored in StreamingScan.scans, not IR children, so prefetch
    # must also handle a SplitScan node passed as the traversal root.
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    path = str(next(tmp_path.glob("*.parquet")))
    split_scans = expand_scan_for_rank(
        _make_parquet_scan([path]),
        IOPartitionPlan(4, IOPartitionFlavor.SPLIT_FILES),
        rank=0,
        nranks=1,
        parquet_options=ParquetOptions(),
    )
    split_scan = split_scans[0]
    assert isinstance(split_scan, SplitScan)

    context = IRExecutionContext()
    prefetch_parquet_file_metadata_for_ir(split_scan, context)

    paths = tuple(split_scan.base_scan.paths)
    assert paths in context.parquet_file_metadata
    assert context.parquet_file_metadata[paths]


def test_prefetch_parquet_file_metadata_no_parquet_scans() -> None:
    context = IRExecutionContext()
    prefetch_parquet_file_metadata_for_ir(Empty({}), context)
    assert context.parquet_file_metadata == {}


# ---------------------------------------------------------------------------
# Tests migrated from tests/streaming/test_scan.py
# ---------------------------------------------------------------------------


def test_split_scan_aligns_to_row_group_boundaries(
    tmp_path, df, streaming_engine_factory
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=1_000),
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=1, row_group_size=10)
    q = pl.scan_parquet(tmp_path)
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize("mask", [None, pl.col("x") < 1_000])
def test_split_scan_predicate(tmp_path, df, mask, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=1_000),
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    q = pl.scan_parquet(tmp_path)
    if mask is not None:
        q = q.filter(mask)
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize("n_files", [2, 3])
@pytest.mark.parametrize("blocksize", [1_000, 10_000, 1_000_000])
def test_target_partition_size(
    tmp_path,
    df,
    blocksize,
    n_files,
    streaming_engine_factory,
    parquet_stats_executor: concurrent.futures.ThreadPoolExecutor,
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=blocksize),
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)
    q = pl.scan_parquet(tmp_path)
    assert_gpu_result_equal(q, engine=streaming_engine)

    # Check partitioning (throwaway engine — no cluster/runtime needed)
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": blocksize},
    )
    qir = Translator(q._ldf.visit(), _engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(_engine)
    ir, info = lower_ir_graph(
        qir,
        config_options,
        collect_statistics(
            qir,
            config_options,
            parquet_stats_executor,
        ),
    )
    count = info[ir].count
    if blocksize <= 12_000:
        assert count > n_files
    else:
        assert count < n_files


def test_scan_join(engine: pl.GPUEngine, tmp_path: Path) -> None:
    # This test exercises some logic on nodes with multiple children (join)
    # where one or more of the children are Scan nodes.
    left = pl.DataFrame({"a": ["a", "b", "c", "d"], "b": [1, 2, 3, 4]})
    right = pl.DataFrame({"a": ["a", "b", "c", "d"], "c": [10, 20, 30, 40]})

    left.write_parquet(tmp_path / "left.parquet")
    right.write_parquet(tmp_path / "right.parquet")

    left_q = pl.scan_parquet(tmp_path / "left.parquet")
    right_q = pl.scan_parquet(tmp_path / "right.parquet")
    q = left_q.join(right_q, on="a", how="inner")
    assert_gpu_result_equal(q, engine=engine)


def test_scan_union(engine: pl.GPUEngine, tmp_path: Path) -> None:
    # This test exercises some logic on nodes with a Union[Scan, ...]
    df = pl.DataFrame({"a": ["a", "b", "c", "d"], "b": [1, 2, 3, 4]})
    df.write_parquet(tmp_path / "data.parquet")

    df_q = pl.scan_parquet(tmp_path / "data.parquet")

    q = pl.concat([df_q, df_q])
    assert_gpu_result_equal(q, engine=engine)


def _make_parquet_scan(paths: list[str]) -> Scan:
    return Scan(
        {"x": DataType(pl.Int64())},
        "parquet",
        {},
        None,
        paths,
        None,
        0,
        -1,
        None,
        None,
        None,
        ParquetOptions(),
    )


@pytest.mark.parametrize(
    "plan,paths,rank,nranks,expected_path_groups",
    [
        (
            IOPartitionPlan(2, IOPartitionFlavor.FUSED_FILES),
            [f"f{i}" for i in range(6)],
            0,
            1,
            [["f0", "f1"], ["f2", "f3"], ["f4", "f5"]],
        ),
        (
            IOPartitionPlan(2, IOPartitionFlavor.FUSED_FILES),
            [f"f{i}" for i in range(6)],
            0,
            2,
            [["f0", "f1"], ["f2", "f3"]],
        ),
        (
            IOPartitionPlan(2, IOPartitionFlavor.FUSED_FILES),
            [f"f{i}" for i in range(6)],
            1,
            2,
            [["f4", "f5"]],
        ),
        (IOPartitionPlan(3, IOPartitionFlavor.SINGLE_READ), ["a", "b", "c"], 1, 2, []),
    ],
)
def test_expand_scan_for_rank_fused_and_single_read(
    plan: IOPartitionPlan,
    paths: list[str],
    rank: int,
    nranks: int,
    expected_path_groups: list[list[str]],
) -> None:
    scans = expand_scan_for_rank(
        _make_parquet_scan(paths),
        plan,
        rank=rank,
        nranks=nranks,
        parquet_options=ParquetOptions(),
    )
    for scan, expected_paths in zip(scans, expected_path_groups, strict=True):
        assert isinstance(scan, Scan)
        assert scan.paths == expected_paths


@pytest.mark.parametrize(
    "rank,expected_splits",
    [
        (0, [(0, 4), (1, 4)]),
        (1, [(2, 4), (3, 4)]),
    ],
)
def test_expand_scan_for_rank_split_files(
    rank: int,
    expected_splits: list[tuple[int, int]],
) -> None:
    plan = IOPartitionPlan(4, IOPartitionFlavor.SPLIT_FILES)
    scans = expand_scan_for_rank(
        _make_parquet_scan(["file.parquet"]),
        plan,
        rank=rank,
        nranks=2,
        parquet_options=ParquetOptions(),
    )
    assert len(scans) == len(expected_splits)
    for scan, (split_index, total_splits) in zip(scans, expected_splits, strict=True):
        assert isinstance(scan, SplitScan)
        assert scan.split_index == split_index
        assert scan.total_splits == total_splits
        assert scan.base_scan.paths == ["file.parquet"]


def test_streaming_scan_raises() -> None:
    # This isn't reachable by normal cudf-polars usage.
    scan = _make_parquet_scan(["file.parquet"])
    ctx = IRExecutionContext()
    with pytest.raises(NotImplementedError, match=r"StreamingScan.do_evaluate"):
        StreamingScan.do_evaluate([scan], scan, context=ctx)


def test_split_scan_do_evaluate_missing_prefetch_metadata() -> None:
    paths = ["/some/missing/file.parquet"]
    parquet_options = ParquetOptions(prefetch_file_metadata=True)
    context = IRExecutionContext()
    schema = {"x": DataType(pl.Int64())}

    with pytest.raises(
        AssertionError,
        match=(
            r"Parquet file metadata was not prefetched for paths: "
            r"\['/some/missing/file\.parquet'\]\."
        ),
    ):
        SplitScan.do_evaluate(
            0,
            4,
            schema,
            "parquet",
            {},
            paths,
            None,
            0,
            -1,
            None,
            None,
            None,
            parquet_options,
            context=context,
        )
