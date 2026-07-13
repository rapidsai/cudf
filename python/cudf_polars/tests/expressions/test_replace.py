# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine


def test_replace(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 2, None, 3]})
    q = df.select(
        pl.col("a").replace([1, 2], [10, 20]).alias("list_replace"),
        pl.col("a").replace(2, 20).alias("scalar_replace"),
        pl.col("a").replace([1, 2], 99).alias("broadcast_replace"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_replace_strict(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 2, None, 3]})
    q = df.select(
        pl.col("a").replace_strict([1, 2], [10, 20], default=-1).alias("list_replace"),
        pl.col("a").replace_strict([1, 2], 99, default=-1).alias("broadcast_replace"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_replace_non_literal_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3], "old": [1, 2, 3]})
    q = df.select(pl.col("a").replace(pl.col("old"), 0))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_replace_strict_without_default_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = df.select(pl.col("a").replace_strict([1, 2], [10, 20]))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_replace_strict_non_literal_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3], "old": [1, 2, 3]})
    q = df.select(pl.col("a").replace_strict(pl.col("old"), 0, default=-1))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("strict", [False, True])
def test_replace_new_length_mismatch(engine: pl.GPUEngine, strict: bool) -> None:  # noqa: FBT001
    df = pl.LazyFrame({"a": [1, 2, 3]})
    if strict:
        expr = pl.col("a").replace_strict([1, 2], [10, 20, 30], default=-1)
    else:
        expr = pl.col("a").replace([1, 2], [10, 20, 30])

    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            df.select(expr).collect(engine=engine)
    else:
        with pytest.raises(
            pl.exceptions.InvalidOperationError,
        ):
            df.select(expr).collect(engine=engine)
