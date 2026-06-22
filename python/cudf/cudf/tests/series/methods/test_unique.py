# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import string

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_datetime_unique(data, nulls):
    rng = np.random.default_rng(seed=0)
    psr = data.copy()

    if len(data) > 0:
        if nulls == "some":
            p = rng.integers(0, len(data), 2)
            psr[p] = None

    gsr = cudf.from_pandas(psr)
    expected = psr.unique()
    got = gsr.unique()

    # Unique does not provide a guarantee on ordering.
    assert_eq(
        pd.Series(expected).sort_values(ignore_index=True),
        got.sort_values(ignore_index=True).to_pandas(),
    )


def test_series_unique():
    rng = np.random.default_rng(seed=0)
    size = 100
    arr = rng.integers(low=-1, high=10, size=size)
    mask = arr != -1
    sr = cudf.Series(arr)
    sr[~mask] = None
    assert set(arr[mask]) == set(sr.unique().dropna().to_numpy())
    assert len(set(arr[mask])) == sr.nunique()


def test_series_nunique(request, nan_as_null, dropna):
    cudf_series = cudf.Series([1, 2, 2, 3, 3], nan_as_null=nan_as_null)
    pd_series = pd.Series([1, 2, 2, 3, 3])
    expect = pd_series.nunique(dropna=dropna)
    got = cudf_series.nunique(dropna=dropna)
    assert expect == got

    cudf_series = cudf.Series(
        [1.0, 2.0, 3.0, np.nan, None], nan_as_null=nan_as_null
    )
    if nan_as_null in {True, None}:
        pd_series = pd.Series([1.0, 2.0, 3.0, None, None])
    else:
        pd_series = pd.Series([1.0, 2.0, 3.0, np.nan, None], dtype=object)

    expect = pd_series.nunique(dropna=dropna)
    got = cudf_series.nunique(dropna=dropna)
    assert expect == got

    cudf_series = cudf.Series([1.0, np.nan, np.nan], nan_as_null=nan_as_null)
    if nan_as_null in {True, None}:
        pd_series = pd.Series([1.0, np.nan, np.nan])
    else:
        pd_series = pd.Series([1.0, None, None])
    expect = pd_series.nunique(dropna=dropna)
    got = cudf_series.nunique(dropna=dropna)
    assert expect == got


@pytest.mark.parametrize(
    "item",
    [
        ["Cbe", "cbe", "CbeD", "Cb", "ghi", "Cb"],
        ["a", "a", "a", "a", "A"],
        ["A"],
        ["abc", "xyz", None, "ab", "123"],
        [None, None, "abc", None, "abc"],
    ],
)
def test_string_unique(item):
    ps = pd.Series(item)
    gs = cudf.Series(item)
    # Pandas `unique` returns a numpy array
    pres = pd.Series(ps.unique())
    # cudf returns a cudf.Series
    gres = gs.unique()
    assert_eq(pres, gres)


def test_categorical_unique():
    num_elements = 20
    rng = np.random.default_rng(seed=12)
    pd_cat = pd.Categorical(
        pd.Series(
            rng.choice(
                list(string.ascii_letters + string.digits), num_elements
            ),
            dtype="category",
        )
    )

    gser = cudf.Series(pd_cat)
    gdf_unique_sorted = np.sort(gser.unique().to_pandas())

    pser = pd.Series(pd_cat)
    pdf_unique_sorted = np.sort(pser.unique())

    np.testing.assert_array_equal(pdf_unique_sorted, gdf_unique_sorted)
