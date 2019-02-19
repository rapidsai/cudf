# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import numpy as np
import pandas as pd

from cudf.dataframe import Series


params_dtypes = [np.int32, np.float32, np.float64]
methods = ['min', 'max', 'sum', 'mean', 'var', 'std']

interpolation_methods = ['linear', 'lower', 'higher', 'midpoint', 'nearest']
exact_results = {
    'linear': [-1.01, 0.3125, 0.7805, 1.62, 6.8],
    'lower': [-1.01, 0.15, 0.15, 1.11, 6.8],
    'higher': [0.15, 0.8, 0.8, 2.13, 6.8],
    'midpoint': [-0.43, 0.475, 0.475, 1.62, 6.8],
    'nearest': [-1.01, 0.15, 0.8, 2.13, 6.8]}
approx_results = [-1.01, 0.15, 0.15, 1.11, 6.8]


@pytest.mark.parametrize('method', methods)
@pytest.mark.parametrize('dtype', params_dtypes)
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
        if method in ['std', 'var']:
            return fn(ddof=1)
        else:
            return fn()

    expect, got = call_test(arr2), call_test(sr)
    print(expect, got)
    np.testing.assert_approx_equal(expect, got)


@pytest.mark.parametrize('method', methods)
def test_series_reductions_concurrency(method):
    from concurrent.futures import ThreadPoolExecutor
    e = ThreadPoolExecutor(10)

    np.random.seed(0)
    srs = [Series(np.random.random(10000)) for _ in range(1)]

    def call_test(sr):
        fn = getattr(sr, method)
        if method in ['std', 'var']:
            return fn(ddof=1)
        else:
            return fn()

    def f(sr):
        return call_test(sr + 1)

    list(e.map(f, srs * 50))


@pytest.mark.parametrize('ddof', range(3))
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
        assert len(set(arr[mask])) == sr.unique_count()
        df = pd.DataFrame(data=arr[mask], columns=['col'])
        expect = df.col.value_counts().sort_index()
        got = sr.value_counts().to_pandas().sort_index()
        print(expect.head())
        print(got.head())
        assert got.equals(expect)


def test_series_scale():
    arr = pd.Series(np.random.randint(low=-10, high=10, size=100))
    sr = Series(arr)

    vmin = arr.min()
    vmax = arr.max()
    scaled = (arr - vmin) / (vmax - vmin)
    assert scaled.min() == 0
    assert scaled.max() == 1
    pd.testing.assert_series_equal(sr.scale().to_pandas(), scaled)


@pytest.mark.parametrize('int_method', interpolation_methods)
def test_exact_quantiles(int_method):
    arr = np.asarray([6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7])
    quant_values = [0.0, 0.25, 0.33, 0.5, 1.0]

    gdf_series = Series(arr)

    q1 = gdf_series.quantile(quant_values, interpolation=int_method,
                             exact=True)

    np.testing.assert_allclose(q1.to_pandas().values,
                               exact_results[int_method], rtol=1e-10)


def test_approx_quantiles():
    arr = np.asarray([6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7])
    quant_values = [0.0, 0.25, 0.33, 0.5, 1.0]

    gdf_series = Series(arr)

    q1 = gdf_series.quantile(quant_values, exact=False)

    np.testing.assert_allclose(q1.to_pandas().values, approx_results,
                               rtol=1e-10)
