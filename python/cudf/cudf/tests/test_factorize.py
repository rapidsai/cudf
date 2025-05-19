# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import DataFrame, Index
from cudf.testing import assert_eq


@pytest.mark.parametrize("ncats,nelem", [(2, 2), (2, 10), (10, 100)])
def test_factorize_series_obj(ncats, nelem):
    df = DataFrame()
    rng = np.random.default_rng(seed=0)

    # initialize data frame
    df["cats"] = arr = rng.integers(2, size=10, dtype=np.int32)

    uvals, labels = df["cats"].factorize()
    unique_values, indices = np.unique(arr, return_index=True)
    expected_values = unique_values[np.argsort(indices)]

    np.testing.assert_array_equal(labels.to_numpy(), expected_values)
    assert isinstance(uvals, cp.ndarray)
    assert isinstance(labels, Index)

    encoder = {labels[idx]: idx for idx in range(len(labels))}
    handcoded = [encoder[v] for v in arr]
    np.testing.assert_array_equal(uvals.get(), handcoded)


@pytest.mark.parametrize("ncats,nelem", [(2, 2), (2, 10), (10, 100)])
def test_factorize_index_obj(ncats, nelem):
    df = DataFrame()
    rng = np.random.default_rng(seed=0)

    # initialize data frame
    df["cats"] = arr = rng.integers(2, size=10, dtype=np.int32)
    df = df.set_index("cats")

    uvals, labels = df.index.factorize()
    unique_values, indices = np.unique(arr, return_index=True)
    expected_values = unique_values[np.argsort(indices)]

    np.testing.assert_array_equal(labels.values.get(), expected_values)
    assert isinstance(uvals, cp.ndarray)
    assert isinstance(labels, Index)

    encoder = {labels[idx]: idx for idx in range(len(labels))}
    handcoded = [encoder[v] for v in arr]
    np.testing.assert_array_equal(uvals.get(), handcoded)


def test_factorize_series_index():
    df = DataFrame()
    df["col1"] = ["C", "H", "C", "W", "W", "W", "W", "W", "C", "W"]
    df["col2"] = [
        2992443.0,
        2992447.0,
        2992466.0,
        2992440.0,
        2992441.0,
        2992442.0,
        2992444.0,
        2992445.0,
        2992446.0,
        2992448.0,
    ]
    assert_eq(df.col1.factorize()[0].get(), df.to_pandas().col1.factorize()[0])
    assert_eq(
        df.col1.factorize()[1].to_pandas().values,
        df.to_pandas().col1.factorize()[1].values,
    )

    df = df.set_index("col2")

    assert_eq(df.col1.factorize()[0].get(), df.to_pandas().col1.factorize()[0])
    assert_eq(
        df.col1.factorize()[1].to_pandas().values,
        df.to_pandas().col1.factorize()[1].values,
    )


def test_cudf_factorize_series():
    data = [1, 2, 3, 4, 5]

    psr = pd.Series(data)
    gsr = cudf.Series(data)

    expect = pd.factorize(psr)
    got = cudf.factorize(gsr)

    assert len(expect) == len(got)

    np.testing.assert_array_equal(expect[0], got[0].get())
    np.testing.assert_array_equal(expect[1], got[1].values.get())


def test_cudf_factorize_index():
    data = [1, 2, 3, 4, 5]

    pi = pd.Index(data)
    gi = cudf.Index(data)

    expect = pd.factorize(pi)
    got = cudf.factorize(gi)

    assert len(expect) == len(got)

    np.testing.assert_array_equal(expect[0], got[0].get())
    np.testing.assert_array_equal(expect[1], got[1].values.get())


def test_cudf_factorize_array():
    data = [1, 2, 3, 4, 5]

    parr = np.array(data)
    garr = cp.array(data)

    expect = pd.factorize(parr)
    got = cudf.factorize(garr)

    assert len(expect) == len(got)

    np.testing.assert_array_equal(expect[0], got[0].get())
    np.testing.assert_array_equal(expect[1], got[1].get())


@pytest.mark.parametrize("pandas_compatibility", [True, False])
def test_factorize_code_pandas_compatibility(pandas_compatibility):
    psr = pd.Series([1, 2, 3, 4, 5])
    gsr = cudf.from_pandas(psr)

    expect = pd.factorize(psr)
    with cudf.option_context("mode.pandas_compatible", pandas_compatibility):
        got = cudf.factorize(gsr)
    assert_eq(got[0], expect[0])
    assert_eq(got[1], expect[1])
    if pandas_compatibility:
        assert got[0].dtype == expect[0].dtype
    else:
        assert got[0].dtype == cudf.dtype("int8")


def test_factorize_result_classes():
    data = [1, 2, 3]

    labels, cats = cudf.factorize(cudf.Series(data))

    assert isinstance(labels, cp.ndarray)
    assert isinstance(cats, cudf.BaseIndex)

    labels, cats = cudf.factorize(cudf.Index(data))

    assert isinstance(labels, cp.ndarray)
    assert isinstance(cats, cudf.BaseIndex)

    labels, cats = cudf.factorize(cp.array(data))

    assert isinstance(labels, cp.ndarray)
    assert isinstance(cats, cp.ndarray)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "def", "abc", "a", "def", None],
        [10, 20, 100, -10, 0, 1, None, 10, 100],
    ],
)
def test_category_dtype_factorize(data):
    gs = cudf.Series(data, dtype="category")
    ps = gs.to_pandas()

    actual_codes, actual_uniques = gs.factorize()
    expected_codes, expected_uniques = ps.factorize()

    assert_eq(actual_codes, expected_codes)
    assert_eq(actual_uniques, expected_uniques)
