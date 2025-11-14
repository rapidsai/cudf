# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_collect_raises,
    assert_gpu_result_equal,
    assert_ir_translation_raises,
    assert_sink_ir_translation_raises,
)


def test_translation_assert_raises():
    df = pl.LazyFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16),
                end=datetime(2021, 12, 16, 3),
                interval="30m",
                eager=True,
            ),
            "n": range(7),
        }
    )

    # This should succeed
    assert_gpu_result_equal(df)

    with pytest.raises(AssertionError):
        # This should fail, because we can translate this query.
        assert_ir_translation_raises(df, NotImplementedError)

    class E(Exception):
        pass

    unsupported = df.group_by_dynamic("time", every="1d").agg(pl.col("n").sum())
    # Unsupported query should raise NotImplementedError
    assert_ir_translation_raises(unsupported, NotImplementedError)

    with pytest.raises(AssertionError):
        # This should fail, because we can't translate this query, but it doesn't raise E.
        assert_ir_translation_raises(unsupported, E)


def test_collect_assert_raises():
    df = pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    with pytest.raises(AssertionError, match="CPU execution DID NOT RAISE"):
        # This should raise, because polars CPU can run this query,
        # but we expect an error.
        assert_collect_raises(
            df,
            polars_except=pl.exceptions.InvalidOperationError,
            cudf_except=(),
        )

    with pytest.raises(AssertionError, match="GPU execution DID NOT RAISE"):
        # This should raise, because polars GPU can run this query,
        # but we expect an error.
        assert_collect_raises(
            df,
            polars_except=(),
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

    with pytest.raises(AssertionError, match="GPU execution RAISED"):
        # This should raise because the expected GPU error is wrong
        assert_collect_raises(
            q,
            polars_except=pl.exceptions.InvalidOperationError,
            cudf_except=NotImplementedError,
        )

    with pytest.raises(AssertionError, match="CPU execution RAISED"):
        # This should raise because the expected CPU error is wrong
        assert_collect_raises(
            q,
            polars_except=NotImplementedError,
            cudf_except=pl.exceptions.InvalidOperationError,
        )


def test_sink_ir_translation_raises_bad_extension():
    df = pl.LazyFrame({"a": [1, 2, 3]})
    # Should raise because ".foo" is not a recognized file extension
    with pytest.raises(ValueError, match="Unsupported file format: .foo"):
        assert_sink_ir_translation_raises(df, Path("out.foo"), {}, NotImplementedError)


def test_sink_ir_translation_raises_sink_error_before_translation(tmp_path: Path):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    # Should raise because "foo" is not a valid write kwarg,
    # so the sink_* function fails before IR translation
    with pytest.raises(
        AssertionError,
        match="Sink function raised an exception before translation: .*foo",
    ):
        assert_sink_ir_translation_raises(
            df, tmp_path / "out.csv", {"foo": True}, NotImplementedError
        )
