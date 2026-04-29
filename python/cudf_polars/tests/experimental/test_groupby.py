# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic GroupBy operations using the rapidsmpf runtime."""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

import polars as pl

from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(scope="module")
def df() -> pl.LazyFrame:
    """Create a test DataFrame for groupby."""
    return pl.LazyFrame(
        {
            "key": list(range(50)) * 3,  # 50 unique keys
            "key2": list(range(10)) * 15,  # 10 unique keys
            "value": range(150),
            "value2": [1.0, 2.0, 3.0, 4.0, 5.0] * 30,
            "x": range(150),
            "xx": list(range(75)) * 2,
            "y": [1, 2, 3] * 50,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 30,
        }
    )


@pytest.mark.parametrize("keys", [("key",), ("key", "key2")])
@pytest.mark.parametrize("agg", ["sum", "mean", "len", "min", "max"])
def test_dynamic_groupby_basic(df, streaming_engine, keys, agg):
    """Test dynamic groupby with various key and agg combinations."""
    expr = getattr(pl.col("value"), agg)()
    q = df.group_by(*keys).agg(expr)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"target_partition_size": 100_000_000}}],
    indirect=True,
)
def test_dynamic_groupby_tree_strategy(df, streaming_engine):
    """Test that small output uses tree reduction (high target_partition_size)."""
    q = df.group_by("key2").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"target_partition_size": 1000}}],
    indirect=True,
)
def test_dynamic_groupby_shuffle_strategy(streaming_engine):
    """Test that large output uses shuffle (low target_partition_size)."""
    df = pl.LazyFrame({"key": range(1000), "value": range(1000)})
    q = df.group_by("key").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_groupby_single_group(streaming_engine):
    """Test dynamic groupby where all rows have the same key."""
    df = pl.LazyFrame({"key": [1] * 100, "value": range(100)})
    q = df.group_by("key").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_groupby_multiple_aggs(df, streaming_engine):
    """Test dynamic groupby with multiple aggregations."""
    q = df.group_by("key").agg(
        pl.col("value").sum().alias("value_sum"),
        pl.col("value").mean().alias("value_mean"),
        pl.col("value2").min().alias("value2_min"),
    )
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_groupby_maintain_order(df, streaming_engine):
    """Test dynamic groupby with maintain_order=True."""
    q = df.group_by("key", maintain_order=True).agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_groupby_single_row(streaming_engine):
    """Test dynamic groupby on single-row DataFrame."""
    df = pl.LazyFrame({"key": [1], "value": [42]})
    q = df.group_by("key").agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


# ---------------------------------------------------------------------------
# Tests migrated from tests/experimental/test_groupby.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "len"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby(df, streaming_engine, op, keys):
    q = getattr(df.group_by(*keys), op)()
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize("op", ["sum", "mean", "len"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"max_rows_per_partition": int(1e9)}}],
    indirect=True,
)
def test_groupby_single_partitions(df, streaming_engine, op, keys):
    q = getattr(df.group_by(*keys), op)()
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "op", ["sum", "mean", "len", "count", "min", "max", "n_unique", "std", "var"]
)
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby_agg(df, streaming_engine, op, keys):
    agg = getattr(pl.col("x"), op)()
    q = df.group_by(*keys).agg(agg)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize("ddof", [0, 2, 50])
@pytest.mark.parametrize("agg", ["std", "var"])
def test_groupby_std_var_ddof(df, engine, agg, ddof):
    q = df.group_by("y").agg(getattr(pl.col("x"), agg)(ddof=ddof))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "fallback_mode,streaming_engine",
    [
        ("silent", {"executor_options": {"fallback_mode": "silent"}}),
        ("raise", {"executor_options": {"fallback_mode": "raise"}}),
        ("warn", {"executor_options": {"fallback_mode": "warn"}}),
        ("foo", {"executor_options": {"fallback_mode": "foo"}}),
    ],
    indirect=["streaming_engine"],
)
def test_groupby_fallback(df, fallback_mode, streaming_engine):
    match = "Failed to decompose groupby aggs"

    q = df.group_by("y").median()

    if fallback_mode == "silent":
        ctx = contextlib.nullcontext()
    elif fallback_mode == "raise":
        ctx = pytest.raises(
            NotImplementedError,
            match=match,
        )
    elif fallback_mode == "foo":
        ctx = pytest.raises(
            pl.exceptions.ComputeError,
            match="'foo' is not a valid StreamingFallbackMode",
        )
    else:
        ctx = pytest.warns(UserWarning, match=match)
    with ctx:
        assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_groupby_agg_literal(df, streaming_engine):
    q = df.group_by("y").agg(1)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "op",
    [
        pl.max("x") - pl.min("x"),
        pl.mean("x") * pl.sum("x"),
        pl.max("x") + pl.max("z"),
        pl.max("x") + 1,
    ],
)
def test_groupby_agg_binop(df: pl.LazyFrame, streaming_engine, op: pl.Expr) -> None:
    q = df.group_by("y").agg(op)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "op, column_name",
    [
        (pl.max("x") - pl.min("x"), "x__max_min"),
        (pl.mean("x"), "x__mean_sum"),
    ],
)
def test_groupby_agg_duplicate(streaming_engine, op: pl.Expr, column_name: str) -> None:
    # Ensure that the column names we create internally don't collide with
    # the user's column names.
    df = pl.LazyFrame(
        {
            "x": [0, 1, 2, 3] * 2,
            column_name: [4, 5, 6, 7] * 2,
            "y": [1, 2, 1, 2] * 2,
        }
    )
    q = df.group_by("y").agg(op, pl.min(column_name))
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_groupby_agg_empty(df: pl.LazyFrame, streaming_engine) -> None:
    q = df.group_by("y").agg()
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.filterwarnings("ignore:This slice not supported for multiple partitions.")
@pytest.mark.parametrize("zlice", [(0, 2), (2, 2), (-2, None)])
def test_groupby_then_slice(streaming_engine, zlice: tuple[int, int]) -> None:
    df = pl.LazyFrame(
        {
            "x": [0, 1, 2, 3] * 2,
            "y": [1, 2, 1, 2] * 2,
        }
    )
    q = df.group_by("y", maintain_order=True).max().slice(*zlice)
    assert_gpu_result_equal(q, engine=streaming_engine)


def test_groupby_on_equality(streaming_engine) -> None:
    # See: https://github.com/rapidsai/cudf/issues/19152
    df = pl.LazyFrame(
        {
            "key1": [1, 1, 1, 2, 3, 1, 4, 6, 7],
            "key2": [2, 2, 2, 2, 6, 1, 4, 6, 8],
            "int32": pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=pl.Int32()),
        }
    )
    q = df.group_by(pl.col("key1") == pl.col("key2")).agg(pl.col("int32").sum())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "values",
    [
        [1, None, 2, None],
        [1, None, None, None],
    ],
)
@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"max_rows_per_partition": 2}}],
    indirect=True,
)
def test_mean_partitioned(values: list[int | None], streaming_engine) -> None:
    df = pl.LazyFrame(
        {
            "key1": [1, 1, 2, 2],
            "uint16_with_null": pl.Series(values, dtype=pl.UInt16()),
        }
    )
    q = df.group_by("key1").agg(pl.col("uint16_with_null").mean())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_groupby_literal_key(df, streaming_engine):
    q = (
        df.group_by(
            pl.lit(True).alias("key"),  # noqa: FBT003
            maintain_order=False,
        )
        .agg(pl.col("x").sum())
        .drop("key")
    )
    assert_gpu_result_equal(q, engine=streaming_engine)


# ---------------------------------------------------------------------------
# Tests migrated from tests/experimental/test_groupby.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "len", "count"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
@pytest.mark.parametrize(
    "streaming_engine",
    [
        {
            "executor_options": {
                "max_rows_per_partition": 4,
                "unique_fraction": {"z": 0.5},
                "groupby_n_ary": 8,
            }
        }
    ],
    indirect=True,
)
def test_groupby_agg_config_options(df, op, keys, streaming_engine):
    agg = getattr(pl.col("x"), op)()
    if op in ("sum", "mean"):
        agg = agg.round(2)  # Unary test coverage
    q = df.group_by(*keys).agg(agg)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"target_partition_size": 1}}],
    indirect=True,
)
def test_groupby_count_type_mismatch(df, streaming_engine):
    q = df.group_by("key", maintain_order=True).agg(pl.col("value").count())
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"target_partition_size": 10, "max_rows_per_partition": 5}}],
    indirect=True,
)
def test_shuffle_reduce_insert_finished_called_on_oom(streaming_engine):
    # Tests that an exception raised inside insert_hash() must not leave the
    # C++ ShufflerAsync without insert_finished() being called.

    def foo(*args, **kwargs):
        raise MemoryError("OOM in insert_hash()")

    df = pl.LazyFrame({"a": range(10), "b": range(10)})
    with (
        patch.object(ShuffleManager.Inserter, "insert_hash", foo),
        pytest.raises(ExceptionGroup) as exc_info,
    ):
        df.group_by("a").agg(pl.col("b").sum()).collect(engine=streaming_engine)
    assert any("OOM in insert_hash" in str(e) for e in exc_info.value.exceptions)
