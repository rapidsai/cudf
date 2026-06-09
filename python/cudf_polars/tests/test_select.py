# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import decimal

import pytest

import polars as pl

from cudf_polars.dsl.ir import IRExecutionContext, Scan
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.config import ParquetOptions


def test_select(engine: pl.GPUEngine):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()

    query = ldf.select(
        pl.col("a") + pl.col("b"), (pl.col("a") * 2 + pl.col("b")).alias("d")
    )

    assert_gpu_result_equal(query, engine=engine)


def test_select_decimal(engine: pl.GPUEngine):
    ldf = pl.LazyFrame(
        {"a": pl.Series(values=[decimal.Decimal("1.0"), None], dtype=pl.Decimal(3, 1))}
    )
    query = ldf.select(pl.col("a"))
    assert_gpu_result_equal(query, engine=engine)


def test_select_decimal_precision_none_result_max_precision():
    ldf = pl.LazyFrame(
        {
            "a": pl.Series(
                values=[decimal.Decimal("1.0"), None], dtype=pl.Decimal(None, 1)
            )
        }
    )
    query = ldf.select(pl.col("a"))
    cpu_result = query.collect()
    gpu_result = query.collect(engine=pl.GPUEngine(executor="in-memory"))
    # See github.com/pola-rs/polars/issues/19784
    # for context on the decimal changes.
    assert cpu_result.schema["a"].precision == 38
    assert gpu_result.schema["a"].precision == 38


def test_select_reduce(engine: pl.GPUEngine):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()

    query = ldf.select(
        (pl.col("a") + pl.col("b")).max(),
        (pl.col("a") * 2 + pl.col("b")).alias("d").mean(),
    )

    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("expr", [pl.col("a").first(), pl.col("a").last()])
def test_select_first_last_empty(engine: pl.GPUEngine, expr):
    ldf = pl.LazyFrame({"a": []}, schema={"a": pl.Int64})
    query = ldf.select(expr)
    assert_gpu_result_equal(query, engine=engine)


def test_select_with_cse_no_agg(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    expr = pl.col("a") + pl.col("a")

    query = df.select(expr, (expr * 2).alias("b"), ((expr * 2) + 10).alias("c"))

    assert_gpu_result_equal(query, engine=engine)


def test_select_with_cse_with_agg(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    expr = pl.col("a") + pl.col("a")
    asum = pl.col("a").sum() + pl.col("a").sum()

    query = df.select(
        expr, (expr * 2).alias("b"), asum.alias("c"), (asum + 10).alias("d")
    )

    assert_gpu_result_equal(query, engine=engine)


def test_select_native_datetime(engine: pl.GPUEngine):
    df = pl.LazyFrame({"c0": [1]})
    query = df.select(pl.datetime(1969, 12, 7, 20, 47, 14))
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("fmt", ["ndjson", "csv"])
def test_select_fast_count_unsupported_formats(engine: pl.GPUEngine, tmp_path, fmt):
    df = pl.DataFrame({"a": [1, 2, 3]})
    file = tmp_path / f"test.{fmt}"
    if fmt == "csv":
        df.write_csv(file)
    elif fmt == "ndjson":
        df.write_ndjson(file)

    q = (
        pl.scan_csv(file).select(pl.len())
        if fmt == "csv"
        else pl.scan_ndjson(file).select(pl.len())
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_select_fast_count_parquet(engine: pl.GPUEngine, tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})
    file = tmp_path / "data.parquet"
    df.write_parquet(file)

    q = pl.scan_parquet(file).select(pl.len())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "zlice",
    [
        (1,),
        (1, 3),
        (-1,),
    ],
)
def test_select_fast_count_parquet_skip_rows(
    engine: pl.GPUEngine, request, tmp_path, zlice
):
    df = pl.DataFrame({"a": [1, 2, 3]})
    file = tmp_path / "data.parquet"
    df.write_parquet(file)

    q = pl.scan_parquet(file).slice(1, 5).select(pl.len())
    assert_gpu_result_equal(q, engine=engine)


PARQUET_FAST_COUNT_ROWS = 10


@pytest.fixture(scope="module")
def parquet_fast_count_df() -> pl.DataFrame:
    return pl.DataFrame({"a": range(PARQUET_FAST_COUNT_ROWS)})


@pytest.fixture
def prefetch_engine() -> pl.GPUEngine:
    return pl.GPUEngine(
        executor="in-memory",
        raise_on_fail=True,
        parquet_options={"prefetch_file_metadata": True},
    )


@pytest.fixture(
    params=[
        pytest.param({"skip_rows": 0, "n_rows": None}, id="all_rows"),
        pytest.param({"skip_rows": 3, "n_rows": None}, id="skip_rows"),
        pytest.param({"skip_rows": 2, "n_rows": 4}, id="skip_rows_and_limit"),
        pytest.param({"skip_rows": 0, "n_rows": 5}, id="n_rows"),
        pytest.param({"skip_rows": 8, "n_rows": 10}, id="skip_near_end"),
        pytest.param(
            {"skip_rows": PARQUET_FAST_COUNT_ROWS, "n_rows": None},
            id="skip_all",
        ),
    ],
)
def parquet_scan_row_bounds(request) -> dict[str, int | None]:
    return request.param


def test_select_fast_count_parquet_prefetch_metadata(
    tmp_path,
    parquet_fast_count_df: pl.DataFrame,
    prefetch_engine: pl.GPUEngine,
    parquet_scan_row_bounds: dict[str, int | None],
) -> None:
    skip_rows = parquet_scan_row_bounds["skip_rows"]
    assert skip_rows is not None
    n_rows = parquet_scan_row_bounds["n_rows"]

    file = tmp_path / "data.parquet"
    parquet_fast_count_df.write_parquet(file)

    if skip_rows == 0 and n_rows is None:
        q = pl.scan_parquet(file)
    elif skip_rows == 0:
        q = pl.scan_parquet(file, n_rows=n_rows)
    elif n_rows is None:
        q = pl.scan_parquet(file).slice(skip_rows)
    else:
        q = pl.scan_parquet(file).slice(skip_rows, n_rows)

    q = q.select(pl.len())
    assert_gpu_result_equal(q, engine=prefetch_engine)


def test_get_parquet_row_count_from_metadata_missing_prefetch() -> None:
    paths = ["/some/missing/file.parquet"]
    parquet_options = ParquetOptions(prefetch_file_metadata=True)
    context = IRExecutionContext()

    with pytest.raises(
        AssertionError,
        match=(
            r"Parquet file metadata was not prefetched for paths: "
            r"\['/some/missing/file\.parquet'\]\."
        ),
    ):
        Scan._get_parquet_row_count_from_metadata(
            paths,
            skip_rows=0,
            n_rows=-1,
            parquet_options=parquet_options,
            context=context,
        )
