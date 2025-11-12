# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import itertools

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq, assert_groupby_results_equal


@pytest.fixture(params=[False, True], ids=["no-null-keys", "null-keys"])
def keys_null(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["no-null-values", "null-values"])
def values_null(request):
    return request.param


@pytest.fixture
def df(keys_null, values_null):
    keys = ["a", "b", "a", "c", "b", "b", "c", "a"]
    r = range(len(keys))
    if keys_null:
        keys[::3] = itertools.repeat(None, len(r[::3]))
    values = list(range(len(keys)))
    if values_null:
        values[1::3] = itertools.repeat(None, len(r[1::3]))
    return cudf.DataFrame({"key": keys, "values": values})


@pytest.mark.parametrize("agg", ["cumsum", "cumprod", "max", "sum", "prod"])
def test_transform_broadcast(agg, df):
    pf = df.to_pandas()
    got = df.groupby("key").transform(agg)
    expect = pf.groupby("key").transform(agg)
    assert_eq(got, expect, check_dtype=False)


def test_transform_invalid():
    df = cudf.DataFrame({"key": [1, 1], "values": [4, 5]})
    with pytest.raises(TypeError):
        df.groupby("key").transform({"values": "cumprod"})


@pytest.mark.parametrize(
    "by",
    [
        "a",
        ["a", "b"],
        pd.Series([2, 1, 1, 2, 2]),
        pd.Series(["b", "a", "a", "b", "b"]),
    ],
)
@pytest.mark.parametrize("agg", ["sum", lambda df: df.mean()])
def test_groupby_transform_aggregation(by, agg):
    gdf = cudf.DataFrame(
        {"a": [2, 2, 1, 2, 1], "b": [1, 1, 1, 2, 2], "c": [1, 2, 3, 4, 5]}
    )
    pdf = gdf.to_pandas()

    expected = pdf.groupby(by).transform(agg)
    actual = gdf.groupby(by).transform(agg)

    assert_groupby_results_equal(expected, actual)


@pytest.mark.parametrize("by", ["a", ["a", "b"], pd.Series([1, 2, 1, 3])])
def test_groupby_transform_maintain_index(by):
    # test that we maintain the index after a groupby transform
    gdf = cudf.DataFrame(
        {"a": [1, 1, 1, 2], "b": [1, 2, 1, 2]}, index=[3, 2, 1, 0]
    )
    pdf = gdf.to_pandas()
    assert_groupby_results_equal(
        pdf.groupby(by).transform("max"), gdf.groupby(by).transform("max")
    )
