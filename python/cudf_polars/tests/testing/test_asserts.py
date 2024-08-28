# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import Select
from cudf_polars.testing.asserts import (
    assert_collect_raises,
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


def test_translation_assert_raises():
    df = pl.LazyFrame({"a": [1, 2, 3]})

    # This should succeed
    assert_gpu_result_equal(df)

    with pytest.raises(AssertionError):
        # This should fail, because we can translate this query.
        assert_ir_translation_raises(df, NotImplementedError)

    class E(Exception):
        pass

    unsupported = df.group_by("a").agg(pl.col("a").upper_bound().alias("b"))
    # Unsupported query should raise NotImplementedError
    assert_ir_translation_raises(unsupported, NotImplementedError)

    with pytest.raises(AssertionError):
        # This should fail, because we can't translate this query, but it doesn't raise E.
        assert_ir_translation_raises(unsupported, E)


def test_collect_assert_raises(monkeypatch):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    with pytest.raises(AssertionError):
        # This should raise, because polars CPU can run this query
        assert_collect_raises(
            df,
            polars_except=pl.exceptions.InvalidOperationError,
            cudf_except=pl.exceptions.InvalidOperationError,
        )

    # Here's an invalid query that gets caught at IR optimisation time.
    q = df.select(pl.col("a") * pl.col("b"))

    # This exception is raised in preprocessing, so is the same for
    # both CPU and GPU engines.
    assert_collect_raises(
        q,
        polars_except=pl.exceptions.InvalidOperationError,
        cudf_except=pl.exceptions.InvalidOperationError,
    )

    with pytest.raises(AssertionError):
        # This should raise because the expected GPU error is wrong
        assert_collect_raises(
            q,
            polars_except=pl.exceptions.InvalidOperationError,
            cudf_except=NotImplementedError,
        )

    with pytest.raises(AssertionError):
        # This should raise because the expected CPU error is wrong
        assert_collect_raises(
            q,
            polars_except=NotImplementedError,
            cudf_except=pl.exceptions.InvalidOperationError,
        )

    with monkeypatch.context() as m:
        m.setattr(Select, "evaluate", lambda self, cache: DataFrame([]))
        # This query should fail, but we monkeypatch a bad
        # implementation of Select which "succeeds" to check that our
        # assertion notices this case.
        q = df.select(pl.col("a") + pl.Series([1, 2]))
        with pytest.raises(AssertionError):
            assert_collect_raises(
                q,
                polars_except=pl.exceptions.ComputeError,
                cudf_except=pl.exceptions.ComputeError,
            )
