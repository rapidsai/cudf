# Copyright (c) 2018-2023, NVIDIA CORPORATION.

from string import ascii_lowercase

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import DataFrame
from cudf.testing import _utils as utils

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
def test_get_dummies(data, index):
    gdf = DataFrame({"x": data}, index=index)
    pdf = pd.DataFrame({"x": data}, index=index)

    encoded_expected = pd.get_dummies(pdf, prefix="test")
    with pytest.warns(FutureWarning):
        encoded_actual = cudf.get_dummies(gdf, prefix="test")

    utils.assert_eq(
        encoded_expected,
        encoded_actual,
        check_dtype=len(data) != 0,
    )
    encoded_actual = cudf.get_dummies(gdf, prefix="test", dtype=np.uint8)

    utils.assert_eq(
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
    with pytest.warns(FutureWarning):
        encoded_actual = cudf.get_dummies(gdf, prefix="test")

    utils.assert_eq(encoded_expected, encoded_actual)


@pytest.mark.parametrize("nan_as_null", [True, False])
@pytest.mark.parametrize("dummy_na", [True, False])
def test_onehost_get_dummies_dummy_na(nan_as_null, dummy_na):
    pdf = pd.DataFrame({"a": [0, 1, np.nan]})
    df = DataFrame.from_pandas(pdf, nan_as_null=nan_as_null)

    expected = pd.get_dummies(pdf, dummy_na=dummy_na, columns=["a"])
    with pytest.warns(FutureWarning):
        got = cudf.get_dummies(df, dummy_na=dummy_na, columns=["a"])

    if dummy_na and nan_as_null:
        got = got.rename(columns={"a_null": "a_nan"})[expected.columns]

    utils.assert_eq(expected, got)


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

    gdf = DataFrame(data)
    pdf = pd.DataFrame(data)

    encoded_expected = pd.get_dummies(
        pdf, prefix=prefix, prefix_sep=prefix_sep
    )
    with pytest.warns(FutureWarning):
        encoded_actual = cudf.get_dummies(
            gdf, prefix=prefix, prefix_sep=prefix_sep
        )

    utils.assert_eq(encoded_expected, encoded_actual)


def test_get_dummies_with_nan():
    df = cudf.DataFrame(
        {"a": cudf.Series([1, 2, np.nan, None], nan_as_null=False)}
    )
    expected = cudf.DataFrame(
        {
            "a_null": [0, 0, 0, 1],
            "a_1.0": [1, 0, 0, 0],
            "a_2.0": [0, 1, 0, 0],
            "a_nan": [0, 0, 1, 0],
        },
        dtype="uint8",
    )
    with pytest.warns(FutureWarning):
        actual = cudf.get_dummies(df, dummy_na=True, columns=["a"])

    utils.assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        cudf.Series(["abc", "l", "a", "abc", "z", "xyz"]),
        cudf.Index([None, 1, 2, 3.3, None, 0.2]),
        cudf.Series([0.1, 2, 3, None, np.nan]),
        cudf.Series([23678, 324, 1, 324], name="abc"),
    ],
)
@pytest.mark.parametrize("prefix_sep", ["-", "#"])
@pytest.mark.parametrize("prefix", [None, "hi"])
@pytest.mark.parametrize("dtype", ["uint8", "int16"])
def test_get_dummies_array_like(data, prefix_sep, prefix, dtype):
    actual = cudf.get_dummies(
        data, prefix=prefix, prefix_sep=prefix_sep, dtype=dtype
    )
    if isinstance(data, (cudf.Series, cudf.BaseIndex)):
        pd_data = data.to_pandas()
    else:
        pd_data = data

    expected = pd.get_dummies(
        pd_data, prefix=prefix, prefix_sep=prefix_sep, dtype=dtype
    )
    utils.assert_eq(expected, actual)


def test_get_dummies_array_like_with_nan():
    ser = cudf.Series([0.1, 2, 3, None, np.nan], nan_as_null=False)
    expected = cudf.DataFrame(
        {
            "a_null": [0, 0, 0, 1, 0],
            "a_0.1": [1, 0, 0, 0, 0],
            "a_2.0": [0, 1, 0, 0, 0],
            "a_3.0": [0, 0, 1, 0, 0],
            "a_nan": [0, 0, 0, 0, 1],
        },
        dtype="uint8",
    )
    with pytest.warns(FutureWarning):
        actual = cudf.get_dummies(
            ser, dummy_na=True, prefix="a", prefix_sep="_"
        )

    utils.assert_eq(expected, actual)
