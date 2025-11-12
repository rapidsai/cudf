# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_assign():
    gdf = cudf.DataFrame({"x": [1, 2, 3]})
    gdf2 = gdf.assign(y=gdf.x + 1)
    assert list(gdf.columns) == ["x"]
    assert list(gdf2.columns) == ["x", "y"]

    np.testing.assert_equal(gdf2.y.to_numpy(), [2, 3, 4])


@pytest.mark.parametrize(
    "mapping",
    [
        {"y": 1, "z": lambda df: df["x"] + df["y"]},
        {
            "x": lambda df: df["x"] * 2,
            "y": lambda df: 2,
            "z": lambda df: df["x"] / df["y"],
        },
    ],
)
def test_assign_callable(mapping):
    df = pd.DataFrame({"x": [1, 2, 3]})
    cdf = cudf.from_pandas(df)
    expect = df.assign(**mapping)
    actual = cdf.assign(**mapping)
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4],
        ["a", "1", "2", "1", "a"],
        pd.Series(["a", "1", "22", "1", "aa"]),
        pd.Series(["a", "1", "22", "1", "aa"], dtype="category"),
        pd.Series([1, 2, 3, 4], dtype="int64"),
        pd.Series([1, 2.3, 3, 4], dtype="float"),
        [None, 1, None, 2, None],
        ["a"],
    ],
)
@pytest.mark.parametrize(
    "categories",
    [
        ["aa", "bb", "cc"],
        [2, 4, 10, 100],
        ["a", "b", "c"],
        ["22", "b", "c"],
        ["a"],
    ],
)
def test_categorical_assignment(data, categories):
    cat_dtype = pd.CategoricalDtype(categories)
    pd_df = pd.DataFrame({"a": np.ones(len(data))})
    cd_df = cudf.from_pandas(pd_df)

    pd_cat_series = pd.Series(data, dtype=cat_dtype)
    # assign categorical series
    pd_df.assign(cat_col=pd_cat_series)
    cd_df.assign(cat_col=pd_cat_series)
    assert_eq(pd_df, cd_df)

    # assign categorical array
    # needed for dask_cudf support for including file name
    # as a categorical column
    # see issue: https://github.com/rapidsai/cudf/issues/2269
    pd_df = pd.DataFrame({"a": np.ones(len(data))})
    cd_df = cudf.from_pandas(pd_df)

    pd_categorical = pd.Categorical(data, dtype=cat_dtype)
    pd_df.assign(cat_col=pd_categorical)
    cd_df.assign(cat_col=pd_categorical)
    assert_eq(pd_df, cd_df)
