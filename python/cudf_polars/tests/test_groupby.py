# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import translate_ir
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def df():
    return pl.LazyFrame(
        {
            "key1": [1, 1, 1, 2, 3, 1, 4, 6, 7],
            "key2": [2, 2, 2, 2, 6, 1, 4, 6, 8],
            "int": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "float": [7.0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )


@pytest.fixture(
    params=[
        ["key1"],
        ["key2"],
        [pl.col("key1") * pl.col("key2")],
        ["key1", "key2"],
        [pl.col("key1") == pl.col("key2")],
        ["key2", pl.col("key1") == pl.lit(1, dtype=pl.Int64)],
    ],
    ids=lambda keys: "-".join(map(str, keys)),
)
def keys(request):
    return request.param


@pytest.fixture(
    params=[
        ["int"],
        ["float", "int"],
        [pl.col("float") + pl.col("int")],
        [pl.col("float").max() - pl.col("int").min()],
        [pl.col("float").mean(), pl.col("int").std()],
        [(pl.col("float") - pl.lit(2)).max()],
    ],
    ids=lambda aggs: "-".join(map(str, aggs)),
)
def exprs(request):
    return request.param


@pytest.fixture(
    params=[
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="Maintaining order in groupby not implemented"
            ),
        ),
    ],
    ids=["no_maintain_order", "maintain_order"],
)
def maintain_order(request):
    return request.param


def test_groupby(df: pl.LazyFrame, maintain_order, keys, exprs):
    q = df.group_by(*keys, maintain_order=maintain_order).agg(*exprs)

    if not maintain_order:
        sort_keys = list(q.schema.keys())[: len(keys)]
        q = q.sort(*sort_keys)

    assert_gpu_result_equal(q, check_exact=False)


def test_groupby_len(df, keys):
    q = df.group_by(*keys).agg(pl.len())

    # TODO: polars returns UInt32, libcudf returns Int32
    with pytest.raises(AssertionError):
        assert_gpu_result_equal(q, check_row_order=False)
    assert_gpu_result_equal(q, check_dtypes=False, check_row_order=False)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("float").is_not_null(),
        (pl.col("int").max() + pl.col("float").min()).max(),
    ],
)
def test_groupby_unsupported(df, expr):
    q = df.group_by("key1").agg(expr)

    with pytest.raises(NotImplementedError):
        _ = translate_ir(q._ldf.visit())
