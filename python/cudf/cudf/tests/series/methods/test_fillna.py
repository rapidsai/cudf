# Copyright (c) 2020-2025, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1, None, 11, 2.0, np.nan],
        [np.nan],
        [None, None, None],
        [np.nan, 1, 10, 393.32, np.nan],
    ],
)
@pytest.mark.parametrize("nan_as_null", [True, False])
@pytest.mark.parametrize("fill_value", [1.2, 332, np.nan])
def test_fillna_with_nan(data, nan_as_null, fill_value):
    gs = cudf.Series(data, dtype="float64", nan_as_null=nan_as_null)
    ps = gs.to_pandas()

    expected = ps.fillna(fill_value)
    actual = gs.fillna(fill_value)

    assert_eq(expected, actual)


def test_fillna_categorical_with_non_categorical_raises():
    ser = cudf.Series([1, None], dtype="category")
    with pytest.raises(TypeError):
        ser.fillna(cudf.Series([1, 2]))


def test_fillna_categorical_with_different_categories_raises():
    ser = cudf.Series([1, None], dtype="category")
    with pytest.raises(TypeError):
        ser.fillna(cudf.Series([1, 2]), dtype="category")


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [12, 12, 22, 343, 4353534, 435342],
        [0.3534, 12, 22, 343, 43.53534, 4353.42],
        np.array([10, 20, 30, 100]),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        np.timedelta64(4, "s"),
        np.timedelta64(456, "D"),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
        "NaT",
    ],
)
def test_timedelta_fillna(data, timedelta_types_as_str, fill_value):
    sr = cudf.Series(data, dtype=timedelta_types_as_str)
    psr = sr.to_pandas()

    expected = psr.dropna()
    actual = sr.dropna()

    assert_eq(expected, actual)

    expected = psr.fillna(fill_value)
    actual = sr.fillna(fill_value)
    assert_eq(expected, actual)

    expected = expected.dropna()
    actual = actual.dropna()

    assert_eq(expected, actual)
