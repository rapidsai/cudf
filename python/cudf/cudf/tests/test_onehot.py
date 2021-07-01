# Copyright (c) 2018, NVIDIA CORPORATION.

from string import ascii_lowercase

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import DataFrame, GenericIndex, Series
from cudf.testing import _utils as utils


def test_onehot_simple():
    np.random.seed(0)
    df = DataFrame()
    # Populate with data [0, 10)
    df["vals"] = np.arange(10, dtype=np.int32)
    # One Hot (Series)
    for i, col in enumerate(df["vals"].one_hot_encoding(list(range(10)))):
        arr = col.to_array()
        # Verify 1 in the right position
        np.testing.assert_equal(arr[i], 1)
        # Every other slots are 0s
        np.testing.assert_equal(arr[:i], 0)
        np.testing.assert_equal(arr[i + 1 :], 0)
    # One Hot (DataFrame)
    df2 = df.one_hot_encoding(
        column="vals", prefix="vals", cats=list(range(10))
    )
    assert df2.columns[0] == "vals"
    for i in range(1, len(df2.columns)):
        assert df2.columns[i] == "vals_%s" % (i - 1)
    got = df2.as_matrix(columns=df2.columns[1:])
    expect = np.identity(got.shape[0])
    np.testing.assert_equal(got, expect)


def test_onehot_random():
    df = DataFrame()
    low = 10
    high = 17
    size = 10
    df["src"] = src = np.random.randint(low=low, high=high, size=size)
    df2 = df.one_hot_encoding(
        column="src", prefix="out_", cats=tuple(range(10, 17))
    )
    mat = df2.as_matrix(columns=df2.columns[1:])

    for val in range(low, high):
        colidx = val - low
        arr = mat[:, colidx]
        mask = src == val
        np.testing.assert_equal(arr, mask)


def test_onehot_masked():
    np.random.seed(0)
    high = 5
    size = 100
    arr = np.random.randint(low=0, high=high, size=size)
    bitmask = utils.random_bitmask(size)
    bytemask = np.asarray(
        utils.expand_bits_to_bytes(bitmask)[:size], dtype=np.bool_
    )
    arr[~bytemask] = -1

    df = DataFrame()
    df["a"] = Series(arr).set_mask(bitmask)

    out = df.one_hot_encoding(
        "a", cats=list(range(high)), prefix="a", dtype=np.int32
    )

    assert tuple(out.columns) == ("a", "a_0", "a_1", "a_2", "a_3", "a_4")
    np.testing.assert_array_equal((out["a_0"] == 1).to_array(), arr == 0)
    np.testing.assert_array_equal((out["a_1"] == 1).to_array(), arr == 1)
    np.testing.assert_array_equal((out["a_2"] == 1).to_array(), arr == 2)
    np.testing.assert_array_equal((out["a_3"] == 1).to_array(), arr == 3)
    np.testing.assert_array_equal((out["a_4"] == 1).to_array(), arr == 4)


def test_onehot_generic_index():
    np.random.seed(0)
    size = 33
    indices = np.random.randint(low=0, high=100, size=size)
    df = DataFrame()
    values = np.random.randint(low=0, high=4, size=size)
    df["fo"] = Series(values, index=GenericIndex(indices))
    out = df.one_hot_encoding(
        "fo", cats=df.fo.unique(), prefix="fo", dtype=np.int32
    )
    assert set(out.columns) == {"fo", "fo_0", "fo_1", "fo_2", "fo_3"}
    np.testing.assert_array_equal(values == 0, out.fo_0.to_array())
    np.testing.assert_array_equal(values == 1, out.fo_1.to_array())
    np.testing.assert_array_equal(values == 2, out.fo_2.to_array())
    np.testing.assert_array_equal(values == 3, out.fo_3.to_array())


@pytest.mark.parametrize(
    "data",
    [
        np.arange(10),
        ["abc", "zyx", "pppp"],
        [],
        pd.Series(["cudf", "hello", "pandas"] * 10, dtype="category"),
    ],
)
def test_get_dummies(data):
    gdf = DataFrame({"x": data})
    pdf = pd.DataFrame({"x": data})

    encoded_expected = pd.get_dummies(pdf, prefix="test")
    encoded_actual = cudf.get_dummies(gdf, prefix="test")

    utils.assert_eq(encoded_expected, encoded_actual)
    encoded_actual = cudf.get_dummies(gdf, prefix="test", dtype=np.uint8)

    utils.assert_eq(encoded_expected, encoded_actual)


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

    utils.assert_eq(encoded_expected, encoded_actual)


@pytest.mark.parametrize("nan_as_null", [True, False])
@pytest.mark.parametrize("dummy_na", [True, False])
def test_onehost_get_dummies_dummy_na(nan_as_null, dummy_na):
    pdf = pd.DataFrame({"a": [0, 1, np.nan]})
    df = DataFrame.from_pandas(pdf, nan_as_null=nan_as_null)

    expected = pd.get_dummies(pdf, dummy_na=dummy_na, columns=["a"])
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
            "a_1.0": [1, 0, 0, 0],
            "a_2.0": [0, 1, 0, 0],
            "a_nan": [0, 0, 1, 0],
            "a_null": [0, 0, 0, 1],
        },
        dtype="uint8",
    )
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
    expected = cudf.get_dummies(
        data, prefix=prefix, prefix_sep=prefix_sep, dtype=dtype
    )
    if isinstance(data, (cudf.Series, cudf.BaseIndex)):
        pd_data = data.to_pandas()
    else:
        pd_data = data

    actual = pd.get_dummies(
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
    actual = cudf.get_dummies(ser, dummy_na=True, prefix="a", prefix_sep="_")

    utils.assert_eq(expected, actual)
