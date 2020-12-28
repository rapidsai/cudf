# Copyright (c) 2018, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import DataFrame, Index
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize("ncats,nelem", [(2, 2), (2, 10), (10, 100)])
def test_factorize_series_obj(ncats, nelem):
    df = DataFrame()
    np.random.seed(0)

    # initialize data frame
    df["cats"] = arr = np.random.randint(2, size=10, dtype=np.int32)

    uvals, labels = df["cats"].factorize()
    np.testing.assert_array_equal(labels.to_array(), sorted(set(arr)))
    assert isinstance(uvals, cp.core.core.ndarray)
    assert isinstance(labels, Index)

    encoder = dict((labels[idx], idx) for idx in range(len(labels)))
    handcoded = [encoder[v] for v in arr]
    np.testing.assert_array_equal(uvals.get(), handcoded)


@pytest.mark.parametrize("ncats,nelem", [(2, 2), (2, 10), (10, 100)])
def test_factorize_index_obj(ncats, nelem):
    df = DataFrame()
    np.random.seed(0)

    # initialize data frame
    df["cats"] = arr = np.random.randint(2, size=10, dtype=np.int32)
    df = df.set_index("cats")

    uvals, labels = df.index.factorize()
    np.testing.assert_array_equal(labels.values.get(), sorted(set(arr)))
    assert isinstance(uvals, cp.core.core.ndarray)
    assert isinstance(labels, Index)

    encoder = dict((labels[idx], idx) for idx in range(len(labels)))
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


def test_factorize_result_classes():
    data = [1, 2, 3]

    labels, cats = cudf.factorize(cudf.Series(data))

    assert isinstance(labels, cp.core.core.ndarray)
    assert isinstance(cats, cudf.Index)

    labels, cats = cudf.factorize(cudf.Index(data))

    assert isinstance(labels, cp.core.core.ndarray)
    assert isinstance(cats, cudf.Index)

    labels, cats = cudf.factorize(cp.array(data))

    assert isinstance(labels, cp.core.core.ndarray)
    assert isinstance(cats, cp.core.core.ndarray)
