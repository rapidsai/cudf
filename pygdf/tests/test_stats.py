import pytest

import numpy as np

from pygdf.dataframe import Series


params_dtypes = [np.int32, np.float32, np.float64]
methods = ['min', 'max', 'sum', 'mean', 'var', 'std']


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
        return fn()

    expect, got = call_test(arr2), call_test(sr)
    print(expect, got)
    np.testing.assert_approx_equal(expect, got)


def test_series_unique():
    for size in [10 ** x for x in range(5)]:
        arr = np.random.randint(low=-1, high=10, size=size)
        mask = arr != -1
        sr = Series.from_masked_array(arr, Series(mask).as_mask())
        assert set(arr[mask]) == set(sr.unique_k(k=10))
    # test out of space
    arr = np.arange(10)
    sr = Series(arr)
    with pytest.raises(ValueError) as raises:
        sr.unique_k(k=7)
    raises.match('too many unique value')


def test_series_scale():
    arr = np.random.randint(low=-10, high=10, size=100)
    sr = Series(arr)

    vmin = arr.min()
    vmax = arr.max()
    scaled = (arr - vmin) / (vmax - vmin)
    assert scaled.min() == 0
    assert scaled.max() == 1
    np.testing.assert_equal(sr.scale().to_array(), scaled)
