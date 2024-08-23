# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
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
