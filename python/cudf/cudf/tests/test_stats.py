# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

from cudf.dataframe import Series

params_dtypes = [np.int32, np.float32, np.float64]
methods = ["min", "max", "sum", "mean", "var", "std"]

interpolation_methods = ["linear", "lower", "higher", "midpoint", "nearest"]


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("dtype", params_dtypes)
def test_series_reductions(method, dtype):
    np.random.seed(0)
    arr = np.random.random(100)
    if np.issubdtype(dtype, np.integer):
        arr *= 100
        mask = arr > 10
    else:
        mask = arr > 0.5

    arr = arr.astype(dtype)
    arr2 = arr[mask]
    sr = Series.from_masked_array(arr, Series(mask).as_mask())

    def call_test(sr):
        fn = getattr(sr, method)
        if method in ["std", "var"]:
            return fn(ddof=1)
        else:
            return fn()

    expect, got = call_test(arr2), call_test(sr)
    print(expect, got)
    np.testing.assert_approx_equal(expect, got)


@pytest.mark.parametrize("method", methods)
def test_series_reductions_concurrency(method):
    from concurrent.futures import ThreadPoolExecutor

    e = ThreadPoolExecutor(10)

    np.random.seed(0)
    srs = [Series(np.random.random(10000)) for _ in range(1)]

    def call_test(sr):
        fn = getattr(sr, method)
        if method in ["std", "var"]:
            return fn(ddof=1)
        else:
            return fn()

    def f(sr):
        return call_test(sr + 1)

    list(e.map(f, srs * 50))


@pytest.mark.parametrize("ddof", range(3))
def test_series_std(ddof):
    np.random.seed(0)
    arr = np.random.random(100) - 0.5
    sr = Series(arr)
    pd = sr.to_pandas()
    got = sr.std(ddof=ddof)
    expect = pd.std(ddof=ddof)
    np.testing.assert_approx_equal(expect, got)


def test_series_unique():
    for size in [10 ** x for x in range(5)]:
        arr = np.random.randint(low=-1, high=10, size=size)
        mask = arr != -1
        sr = Series.from_masked_array(arr, Series(mask).as_mask())
        assert set(arr[mask]) == set(sr.unique().to_array())
        assert len(set(arr[mask])) == sr.nunique()
        df = pd.DataFrame(data=arr[mask], columns=["col"])
        expect = df.col.value_counts().sort_index()
        got = sr.value_counts().to_pandas().sort_index()
        print(expect.head())
        print(got.head())
        assert got.equals(expect)


@pytest.mark.parametrize(
    "nan_as_null, dropna",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_series_nunique(nan_as_null, dropna):
    # We remove nulls as opposed to NaNs using the dropna parameter,
    # so to test against pandas we replace NaN with another discrete value
    cudf_series = Series([1, 2, 2, 3, 3], nan_as_null=nan_as_null)
    pd_series = pd.Series([1, 2, 2, 3, 3])
    expect = pd_series.nunique(dropna=dropna)
    got = cudf_series.nunique(dropna=dropna)
    assert expect == got

    cudf_series = Series(
        [1.0, 2.0, 3.0, np.nan, None], nan_as_null=nan_as_null
    )
    if nan_as_null is True:
        pd_series = pd.Series([1.0, 2.0, 3.0, np.nan, None])
    else:
        pd_series = pd.Series([1.0, 2.0, 3.0, -1.0, None])

    expect = pd_series.nunique(dropna=dropna)
    got = cudf_series.nunique(dropna=dropna)
    assert expect == got

    cudf_series = Series([1.0, np.nan, np.nan], nan_as_null=nan_as_null)
    if nan_as_null is True:
        pd_series = pd.Series([1.0, np.nan, np.nan])
    else:
        pd_series = pd.Series([1.0, -1.0, -1.0])
    expect = pd_series.nunique(dropna=dropna)
    got = cudf_series.nunique(dropna=dropna)
    assert expect == got


def test_series_scale():
    arr = pd.Series(np.random.randint(low=-10, high=10, size=100))
    sr = Series(arr)

    vmin = arr.min()
    vmax = arr.max()
    scaled = (arr - vmin) / (vmax - vmin)
    assert scaled.min() == 0
    assert scaled.max() == 1
    pd.testing.assert_series_equal(sr.scale().to_pandas(), scaled)


@pytest.mark.parametrize("int_method", interpolation_methods)
def test_exact_quantiles(int_method):
    arr = np.asarray([6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7])
    quant_values = [0.0, 0.25, 0.33, 0.5, 1.0]

    df = pd.DataFrame(arr)
    gdf_series = Series(arr)

    q1 = gdf_series.quantile(
        quant_values, interpolation=int_method, exact=True
    )

    q2 = df.quantile(quant_values, interpolation=int_method)

    np.testing.assert_allclose(
        q1.to_pandas().values, np.array(q2.values).T.flatten(), rtol=1e-10
    )


@pytest.mark.parametrize("int_method", interpolation_methods)
def test_exact_quantiles_int(int_method):
    arr = np.asarray([7, 0, 3, 4, 2, 1, -1, 1, 6])
    quant_values = [0.0, 0.25, 0.33, 0.5, 1.0]

    df = pd.DataFrame(arr)
    gdf_series = Series(arr)

    q1 = gdf_series.quantile(
        quant_values, interpolation=int_method, exact=True
    )

    q2 = df.quantile(quant_values, interpolation=int_method)

    np.testing.assert_allclose(
        q1.to_pandas().values, np.array(q2.values).T.flatten(), rtol=1e-10
    )


def test_approx_quantiles():
    arr = np.asarray([6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7])
    quant_values = [0.0, 0.25, 0.33, 0.5, 1.0]
    approx_results = [-1.01, 0.8, 0.8, 2.13, 6.8]

    gdf_series = Series(arr)

    q1 = gdf_series.quantile(quant_values, exact=False)

    np.testing.assert_allclose(
        q1.to_pandas().values, approx_results, rtol=1e-10
    )


def test_approx_quantiles_int():
    arr = np.asarray([1, 2, 3])
    quant_values = [0.5]
    approx_results = [2]

    gdf_series = Series(arr)

    q1 = gdf_series.quantile(quant_values, exact=False)

    assert approx_results == q1.to_pandas().values


@pytest.mark.parametrize("data", [[], [1, 2, 3, 10, 326497]])
@pytest.mark.parametrize("q", [[], 0.5, 1, 0.234, [0.345], [0.243, 0.5, 1]])
def test_misc_quantiles(data, q):
    from cudf.tests import utils

    pdf_series = pd.Series(data)
    gdf_series = Series(data)

    expected = pdf_series.quantile(q)
    actual = gdf_series.quantile(q)
    utils.assert_eq(expected, actual)
