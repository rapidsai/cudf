# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.base import SerializedDataSourceInfo, StatsCollector
from cudf_polars.experimental.io import (
    DataFrameSourceInfo,
    ParquetSourceInfo,
    _clear_source_info_cache,
)
from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.io import make_lazy_frame, make_partitioned_source
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    import pathlib


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "x": range(3_000),
            "y": ["cat", "dog", "fish"] * 1_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 600,
        }
    )


# Simple engine for IR translation / stats collection only (no actual GPU execution)
@pytest.fixture(scope="module")
def stats_engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 10_000,
            "max_rows_per_partition": 1_000,
        },
    )


def test_base_stats_dataframescan(df, stats_engine):
    row_count = df.height
    q = pl.LazyFrame(df)
    ir = Translator(q._ldf.visit(), stats_engine).translate_ir()
    stats = collect_statistics(ir, ConfigOptions.from_polars_engine(stats_engine))

    source = stats.scan_stats[ir]
    assert source.row_count == row_count

    # Storage size is not available for in-memory frames
    assert source.column_storage_size("x") is None


@pytest.mark.parametrize("n_files", [1, 3])
@pytest.mark.parametrize("row_group_size", [None, 10_000])
@pytest.mark.parametrize("max_footer_samples", [3, 0])
@pytest.mark.parametrize("max_row_group_samples", [1, 0])
def test_base_stats_parquet(
    tmp_path,
    df,
    n_files,
    row_group_size,
    max_footer_samples,
    max_row_group_samples,
):
    _clear_source_info_cache()
    make_partitioned_source(
        df,
        tmp_path,
        "parquet",
        n_files=n_files,
        row_group_size=row_group_size,
    )
    q = pl.scan_parquet(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": 10_000},
        parquet_options={
            "max_footer_samples": max_footer_samples,
            "max_row_group_samples": max_row_group_samples,
        },
    )
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_statistics(ir, ConfigOptions.from_polars_engine(engine))
    source = stats.scan_stats[ir]

    if max_footer_samples:
        assert source.row_count == df.height
        assert source.column_storage_size("x") is not None
        assert source.column_storage_size("x") > 0
        assert source.column_storage_size("y") is not None
        assert source.column_storage_size("y") > 0
    else:
        assert source.row_count is None
        assert source.column_storage_size("x") is None
        assert source.column_storage_size("y") is None


def test_dataframescan_stats_pickle(stats_engine):
    df = pl.DataFrame({"x": range(100), "y": [1, 2] * 50})
    q = pl.LazyFrame(df)
    ir = Translator(q._ldf.visit(), stats_engine).translate_ir()
    stats = collect_statistics(ir, ConfigOptions.from_polars_engine(stats_engine))

    # Pickle and unpickle the stats collector
    pickled = pickle.dumps(stats)
    unpickled_stats = pickle.loads(pickled)

    # Verify the unpickled stats are equivalent
    assert type(unpickled_stats) is type(stats)
    assert unpickled_stats.scan_stats[ir].row_count == 100


def test_parquet_round_trip() -> None:
    info = ParquetSourceInfo(1000, {"x": 200, "y": 400})
    data = info.serialize()
    restored = ParquetSourceInfo.deserialize(data)

    assert restored.type == "parquet"
    assert restored.row_count == info.row_count
    assert restored.per_file_means == info.per_file_means


def test_parquet_round_trip_empty() -> None:
    info = ParquetSourceInfo(None, {})
    data = info.serialize()
    restored = ParquetSourceInfo.deserialize(data)

    assert restored.row_count is None
    assert restored.per_file_means == {}


def test_dataframe_round_trip() -> None:
    info = DataFrameSourceInfo(2500)
    data = info.serialize()
    restored = DataFrameSourceInfo.deserialize(data)

    assert restored.type == "dataframe"
    assert restored.row_count == info.row_count
    assert restored.column_storage_size("x") is None


@pytest.mark.parametrize("kind", ["parquet", "csv", "frame"])
def test_stats_planning(tmp_path, kind, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=10_000, max_rows_per_partition=1_000),
    )
    sales = pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "customer_id": [101, 102, 101, 103, 102, 101],
            "amount": [50.0, 75.0, 30.0, 120.0, 85.0, 40.0],
            "product": ["A", "B", "A", "C", "B", "A"],
        }
    )
    sales = make_lazy_frame(sales, kind, path=tmp_path / f"sales_{kind}")
    customers = pl.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "customer_name": ["Alice", "Bob", "Charlie"],
            "region": ["North", "South", "North"],
        }
    )
    customers = make_lazy_frame(customers, kind, path=tmp_path / f"customers_{kind}")
    q_join = sales.filter(pl.col("amount") < 100.0).join(
        customers, on="customer_id", how="inner"
    )
    q_gb = q_join.group_by("customer_id").agg(
        [
            pl.col("amount").sum().alias("total_amount"),
            pl.col("order_id").count().alias("order_count"),
            pl.col("customer_name").first().alias("name"),
            pl.col("region").first().alias("region"),
        ]
    )
    assert_gpu_result_equal(q_gb.sort("customer_id"), engine=streaming_engine)


def test_parquet_deserialize_wrong_type() -> None:
    data = DataFrameSourceInfo(100).serialize()
    with pytest.raises(ValueError, match="Expected ParquetSourceInfo"):
        ParquetSourceInfo.deserialize(data)


def test_dataframe_deserialize_wrong_type() -> None:
    data = ParquetSourceInfo(1000, {"x": 200}).serialize()
    with pytest.raises(ValueError, match="Expected DataFrameSourceInfo"):
        DataFrameSourceInfo.deserialize(data)


def test_dataframe_deserialize_missing_row_count() -> None:
    data = SerializedDataSourceInfo(
        type="dataframe", row_count=None, per_file_means=None
    )
    with pytest.raises(
        ValueError, match="Row count is required for DataFrameSourceInfo"
    ):
        DataFrameSourceInfo.deserialize(data)


def test_parquet_empty_per_file_means() -> None:
    per_file_means: dict[str, int] = {}
    info = ParquetSourceInfo(1000, per_file_means)
    assert info.per_file_means is per_file_means

    info = ParquetSourceInfo(1000, None)
    assert info.per_file_means == {}


def test_serialize_stats_roundtrip_dataframescan(stats_engine: pl.GPUEngine) -> None:
    df = pl.DataFrame({"x": range(200), "y": [1, 2] * 100})
    q = pl.LazyFrame(df)
    ir = Translator(q._ldf.visit(), stats_engine).translate_ir()
    config = ConfigOptions.from_polars_engine(stats_engine)
    stats = collect_statistics(ir, config)

    serialized = stats.serialize(ir)
    wire = json.loads(json.dumps(serialized))
    restored = StatsCollector.deserialize(wire, ir)

    assert set(restored.scan_stats) == set(stats.scan_stats)
    for node, info in stats.scan_stats.items():
        rt = restored.scan_stats[node]
        assert rt.row_count == info.row_count
        assert rt.column_storage_size("x") == info.column_storage_size("x")


def test_serialize_stats_roundtrip_parquet(
    tmp_path: pathlib.Path, df: pl.DataFrame
) -> None:
    _clear_source_info_cache()
    make_partitioned_source(df, tmp_path, "parquet", n_files=3)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": 10_000},
        parquet_options={"max_footer_samples": 3, "max_row_group_samples": 1},
    )
    q = pl.scan_parquet(tmp_path)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    config = ConfigOptions.from_polars_engine(engine)
    stats = collect_statistics(ir, config)

    serialized = stats.serialize(ir)
    wire = json.loads(json.dumps(serialized))
    restored = StatsCollector.deserialize(wire, ir)

    assert set(restored.scan_stats) == set(stats.scan_stats)
    for node, info in stats.scan_stats.items():
        rt = restored.scan_stats[node]
        assert rt.row_count == info.row_count
        for col in ("x", "y", "z"):
            assert rt.column_storage_size(col) == info.column_storage_size(col)
