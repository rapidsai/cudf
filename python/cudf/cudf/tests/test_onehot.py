# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from string import ascii_lowercase

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq

pytestmark = pytest.mark.spilling


@pytest.mark.parametrize(
    "data, index",
    [
        (np.arange(10), None),
        (["abc", "zyx", "pppp"], None),
        ([], None),
        (pd.Series(["cudf", "hello", "pandas"] * 10, dtype="category"), None),
        (range(10), [1, 2, 3, 4, 5] * 2),
    ],
)
@pytest.mark.parametrize("dtype", ["bool", "uint8"])
def test_get_dummies(data, index, dtype):
    pdf = pd.DataFrame({"x": data}, index=index)
    gdf = cudf.from_pandas(pdf)

    encoded_expected = pd.get_dummies(pdf, prefix="test", dtype=dtype)
    encoded_actual = cudf.get_dummies(gdf, prefix="test", dtype=dtype)

    assert_eq(
        encoded_expected,
        encoded_actual,
        check_dtype=len(data) != 0,
    )


@pytest.mark.parametrize("n_cols", [5, 10, 20])
def test_onehot_get_dummies_multicol(n_cols):
    n_categories = 5
    data = dict(
        zip(ascii_lowercase, (np.arange(n_categories) for _ in range(n_cols)))
    )

    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    encoded_expected = pd.get_dummies(pdf, prefix="test")
    encoded_actual = cudf.get_dummies(gdf, prefix="test")

    assert_eq(encoded_expected, encoded_actual)


@pytest.mark.parametrize("nan_as_null", [True, False])
@pytest.mark.parametrize("dummy_na", [True, False])
def test_onehost_get_dummies_dummy_na(nan_as_null, dummy_na):
    df = cudf.DataFrame({"a": [0, 1, np.nan]}, nan_as_null=nan_as_null)
    pdf = df.to_pandas(nullable=nan_as_null)

    expected = pd.get_dummies(pdf, dummy_na=dummy_na, columns=["a"])
    got = cudf.get_dummies(df, dummy_na=dummy_na, columns=["a"])

    assert_eq(expected, got, check_like=True)


@pytest.mark.parametrize(
    "prefix",
    [
        ["a", "b", "c"],
        "",
        None,
        {"first": "one", "second": "two", "third": "three"},
        "--",
    ],
)
@pytest.mark.parametrize(
    "prefix_sep",
    [
        ["a", "b", "c"],
        "",
        "++",
        {"first": "*******", "second": "__________", "third": "#########"},
    ],
)
def test_get_dummies_prefix_sep(prefix, prefix_sep):
    data = {
        "first": ["1", "2", "3"],
        "second": ["abc", "def", "ghi"],
        "third": ["ji", "ji", "ji"],
    }

    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    encoded_expected = pd.get_dummies(
        pdf, prefix=prefix, prefix_sep=prefix_sep
    )
    encoded_actual = cudf.get_dummies(
        gdf, prefix=prefix, prefix_sep=prefix_sep
    )

    assert_eq(encoded_expected, encoded_actual)


def test_get_dummies_with_nan():
    df = cudf.DataFrame(
        {"a": cudf.Series([1, 2, np.nan, None], nan_as_null=False)}
    )

    expected = pd.get_dummies(
        df.to_pandas(nullable=True), dummy_na=True, columns=["a"]
    )

    actual = cudf.get_dummies(df, dummy_na=True, columns=["a"])

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        lambda: cudf.Series(["abc", "l", "a", "abc", "z", "xyz"]),
        lambda: cudf.Index([None, 1, 2, 3.3, None, 0.2]),
        lambda: cudf.Series([0.1, 2, 3, None, np.nan]),
        lambda: cudf.Series([23678, 324, 1, 324], name="abc"),
    ],
)
@pytest.mark.parametrize("prefix_sep", ["-", "#"])
@pytest.mark.parametrize("prefix", [None, "hi"])
@pytest.mark.parametrize("dtype", ["uint8", "int16"])
def test_get_dummies_array_like(data, prefix_sep, prefix, dtype):
    data = data()
    pd_data = data.to_pandas()

    expected = pd.get_dummies(
        pd_data, prefix=prefix, prefix_sep=prefix_sep, dtype=dtype
    )

    actual = cudf.get_dummies(
        data, prefix=prefix, prefix_sep=prefix_sep, dtype=dtype
    )

    assert_eq(expected, actual)


def test_get_dummies_array_like_with_nan():
    ser = cudf.Series([0.1, 2, 3, None, np.nan], nan_as_null=False)

    expected = pd.get_dummies(
        ser.to_pandas(nullable=True), dummy_na=True, prefix="a", prefix_sep="_"
    )

    actual = cudf.get_dummies(ser, dummy_na=True, prefix="a", prefix_sep="_")

    assert_eq(expected, actual)
