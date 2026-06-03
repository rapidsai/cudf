# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl.ir import Scan
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.base import IOPartitionFlavor, IOPartitionPlan
from cudf_polars.streaming.io import SplitScan, expand_scan_for_rank
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
    "plan,paths,rank,nranks,expected_len",
    [
        (
            IOPartitionPlan(2, IOPartitionFlavor.FUSED_FILES),
            [f"f{i}" for i in range(6)],
            0,
            1,
            3,
        ),
        (
            IOPartitionPlan(2, IOPartitionFlavor.FUSED_FILES),
            [f"f{i}" for i in range(6)],
            0,
            2,
            2,
        ),
        (
            IOPartitionPlan(2, IOPartitionFlavor.FUSED_FILES),
            [f"f{i}" for i in range(6)],
            1,
            2,
            1,
        ),
        (IOPartitionPlan(3, IOPartitionFlavor.SINGLE_READ), ["a", "b", "c"], 1, 2, 0),
    ],
)
def test_expand_scan_for_rank_fused_and_single_read(
    plan: IOPartitionPlan,
    paths: list[str],
    rank: int,
    nranks: int,
    expected_len: int,
) -> None:
    scans = expand_scan_for_rank(
        _make_parquet_scan(paths),
        plan,
        rank=rank,
        nranks=nranks,
        parquet_options=ParquetOptions(),
    )
    assert len(scans) == expected_len
    assert all(not isinstance(scan, SplitScan) for scan in scans)


def test_expand_scan_for_rank_split_files() -> None:
    plan = IOPartitionPlan(4, IOPartitionFlavor.SPLIT_FILES)
    scans = expand_scan_for_rank(
        _make_parquet_scan(["file.parquet"]),
        plan,
        rank=0,
        nranks=2,
        parquet_options=ParquetOptions(),
    )
    assert len(scans) == 2
    assert all(isinstance(scan, SplitScan) for scan in scans)


def test_expand_scan_for_rank_treats_zero_nranks_as_one() -> None:
    plan = IOPartitionPlan(1, IOPartitionFlavor.FUSED_FILES)
    scans = expand_scan_for_rank(
        _make_parquet_scan(["a", "b"]),
        plan,
        rank=0,
        nranks=0,
        parquet_options=ParquetOptions(),
    )
    assert len(scans) == 2
