# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pytest

import cudf


@pytest.mark.parametrize("null_flag", [False, True])
def test_kurtosis_df(null_flag, numeric_only):
    data = cudf.DataFrame(
        {
            "a": np.arange(10, dtype="float64"),
            "b": np.arange(10, dtype="int64"),
            "c": np.arange(10, dtype="float64"),
            "d": ["a"] * 10,
        }
    )
    if not numeric_only:
        data = data.select_dtypes(include="number")
    pdata = data.to_pandas()

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.kurtosis(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()

    expected = pdata.kurtosis(numeric_only=numeric_only)
    np.testing.assert_array_almost_equal(got, expected)

    got = data.kurt(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()

    expected = pdata.kurt(numeric_only=numeric_only)
    np.testing.assert_array_almost_equal(got, expected)


@pytest.mark.parametrize("null_flag", [False, True])
def test_skew_df(null_flag, numeric_only):
    data = cudf.DataFrame(
        {
            "a": np.arange(10, dtype="float64"),
            "b": np.arange(10, dtype="int64"),
            "c": np.arange(10, dtype="float64"),
            "d": ["a"] * 10,
        }
    )
    if not numeric_only:
        data = data.select_dtypes(include="number")
    pdata = data.to_pandas()

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.skew(numeric_only=numeric_only)
    expected = pdata.skew(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()
    np.testing.assert_array_almost_equal(got, expected)
