# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1.0, 2, None, 4],
        ["one", "two", "three", "four"],
        pd.Series(["a", "b", "c", "d"], dtype="category"),
        pd.Series(pd.date_range("2010-01-01", "2010-01-04")),
    ],
)
@pytest.mark.parametrize("nulls", ["one", "some", "all", "none"])
def test_dropna_series(data, nulls, inplace):
    psr = pd.Series(data)
    rng = np.random.default_rng(seed=0)
    if len(data) > 0:
        if nulls == "one":
            p = rng.integers(0, 4)
            psr[p] = None
        elif nulls == "some":
            p1, p2 = rng.integers(0, 4, (2,))
            psr[p1] = None
            psr[p2] = None
        elif nulls == "all":
            psr[:] = None

    gsr = cudf.from_pandas(psr)

    check_dtype = True
    if gsr.null_count == len(gsr):
        check_dtype = False

    expected = psr.dropna()
    actual = gsr.dropna()

    if inplace:
        expected = psr
        actual = gsr

    assert_eq(expected, actual, check_dtype=check_dtype)


def test_dropna_nan_as_null():
    sr = cudf.Series([1.0, 2.0, np.nan, None], nan_as_null=False)
    assert_eq(sr.dropna(), sr[:2])
    sr = sr.nans_to_nulls()
    assert_eq(sr.dropna(), sr[:2])

    df = cudf.DataFrame(
        {
            "a": cudf.Series([1.0, 2.0, np.nan, None], nan_as_null=False),
            "b": cudf.Series([1, 2, 3, 4]),
        }
    )

    got = df.dropna()
    expected = df[:2]
    assert_eq(expected, got)

    df = df.nans_to_nulls()
    got = df.dropna()
    expected = df[:2]
    assert_eq(expected, got)


def test_ignore_index():
    pser = pd.Series([1, 2, np.nan], index=[2, 4, 1])
    gser = cudf.from_pandas(pser)

    result = pser.dropna(ignore_index=True)
    expected = gser.dropna(ignore_index=True)
    assert_eq(result, expected)
