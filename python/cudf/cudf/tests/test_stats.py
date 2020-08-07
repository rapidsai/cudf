# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

from cudf.core import Series
from cudf.datasets import randomdata
from cudf.tests.utils import assert_eq

params_dtypes = [np.int32, np.uint32, np.float32, np.float64]
methods = ["min", "max", "sum", "mean", "var", "std"]

interpolation_methods = ["linear", "lower", "higher", "midpoint", "nearest"]


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("dtype", params_dtypes)
@pytest.mark.parametrize("skipna", [True, False])
def test_series_reductions(method, dtype, skipna):
    np.random.seed(0)
    arr = np.random.random(100)
    if np.issubdtype(dtype, np.integer):
        arr *= 100
        mask = arr > 10
    else:
        mask = arr > 0.5

    arr = arr.astype(dtype)
    if dtype in (np.float32, np.float64):
        arr[[2, 5, 14, 19, 50, 70]] = np.nan
    sr = Series.from_masked_array(arr, Series(mask).as_mask())
    psr = sr.to_pandas(nullable_pd_dtype=False)
    psr[~mask] = np.nan

    def call_test(sr, skipna):
        fn = getattr(sr, method)
        if method in ["std", "var"]:
            return fn(ddof=1, skipna=skipna)
        else:
            return fn(skipna=skipna)

    expect, got = call_test(psr, skipna=skipna), call_test(sr, skipna=skipna)
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
    assert_eq(sr.scale(), scaled)


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

    gdf_series = Series(arr)
    pdf_series = pd.Series(arr)

    q1 = gdf_series.quantile(quant_values, exact=False)
    q2 = pdf_series.quantile(quant_values)

    assert_eq(q1, q2)


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

    pdf_series = pd.Series(data)
    gdf_series = Series(data)

    expected = pdf_series.quantile(q)
    actual = gdf_series.quantile(q)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        Series(np.random.normal(-100, 100, 1000)),
        Series(np.random.randint(-50, 50, 1000)),
        Series(np.zeros(100)),
        Series(np.repeat(np.nan, 100)),
        Series(np.array([1.123, 2.343, np.nan, 0.0])),
        Series(
            [5, 10, 53, None, np.nan, None, 12, 43, -423], nan_as_null=False
        ),
        Series([1.1032, 2.32, 43.4, 13, -312.0], index=[0, 4, 3, 19, 6]),
        Series([]),
        Series([-3]),
        randomdata(
            nrows=1000, dtypes={"a": float, "b": int, "c": float, "d": str}
        ),
    ],
)
@pytest.mark.parametrize("null_flag", [False, True])
def test_kurtosis(data, null_flag):
    pdata = data.to_pandas(nullable_pd_dtype=False)

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.kurtosis()
    got = got if np.isscalar(got) else got.to_array()
    expected = pdata.kurtosis()
    np.testing.assert_array_almost_equal(got, expected)

    got = data.kurt()
    got = got if np.isscalar(got) else got.to_array()
    expected = pdata.kurt()
    np.testing.assert_array_almost_equal(got, expected)

    with pytest.raises(NotImplementedError):
        data.kurt(numeric_only=False)


@pytest.mark.parametrize(
    "data",
    [
        Series(np.random.normal(-100, 100, 1000)),
        Series(np.random.randint(-50, 50, 1000)),
        Series(np.zeros(100)),
        Series(np.repeat(np.nan, 100)),
        Series(np.array([1.123, 2.343, np.nan, 0.0])),
        Series(
            [5, 10, 53, None, np.nan, None, 12, 43, -423], nan_as_null=False
        ),
        Series([1.1032, 2.32, 43.4, 13, -312.0], index=[0, 4, 3, 19, 6]),
        Series([]),
        Series([-3]),
        randomdata(
            nrows=1000, dtypes={"a": float, "b": int, "c": float, "d": str}
        ),
    ],
)
@pytest.mark.parametrize("null_flag", [False, True])
def test_skew(data, null_flag):
    pdata = data.to_pandas(nullable_pd_dtype=False)

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.skew()
    expected = pdata.skew()
    got = got if np.isscalar(got) else got.to_array()
    np.testing.assert_array_almost_equal(got, expected)

    with pytest.raises(NotImplementedError):
        data.skew(numeric_only=False)


@pytest.mark.parametrize("dtype", params_dtypes)
@pytest.mark.parametrize("num_na", [0, 1, 50, 99, 100])
def test_series_median(dtype, num_na):
    np.random.seed(0)
    arr = np.random.random(100)
    if np.issubdtype(dtype, np.integer):
        arr *= 100
    mask = np.arange(100) >= num_na

    arr = arr.astype(dtype)
    sr = Series.from_masked_array(arr, Series(mask).as_mask())
    arr2 = arr[mask]
    ps = pd.Series(arr2, dtype=dtype)

    actual = sr.median(skipna=True)
    desired = ps.median(skipna=True)
    print(actual, desired)
    np.testing.assert_approx_equal(actual, desired)

    # only for float until integer null supported convert to pandas in cudf
    # eg. pd.Int64Dtype
    if np.issubdtype(dtype, np.floating):
        ps = sr.to_pandas()
        actual = sr.median(skipna=False)
        desired = ps.median(skipna=False)
        np.testing.assert_approx_equal(actual, desired)


@pytest.mark.parametrize(
    "data1",
    [
        np.random.normal(-100, 100, 1000),
        np.random.randint(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        Series([5, 10, 53, None, np.nan, None], nan_as_null=False),
        Series([1.1, 2.32, 43.4], index=[0, 4, 3]),
        Series([]),
        Series([-3]),
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        np.random.normal(-100, 100, 1000),
        np.random.randint(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        Series([1.1, 2.32, 43.4], index=[0, 500, 4000]),
        Series([5]),
    ],
)
def test_cov1d(data1, data2):
    gs1 = Series(data1)
    gs2 = Series(data2)

    ps1 = gs1.to_pandas(nullable_pd_dtype=False)
    ps2 = gs2.to_pandas(nullable_pd_dtype=False)

    got = gs1.cov(gs2)
    expected = ps1.cov(ps2)
    np.testing.assert_approx_equal(got, expected, significant=8)


@pytest.mark.parametrize(
    "data1",
    [
        np.random.normal(-100, 100, 1000),
        np.random.randint(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        Series([5, 10, 53, None, np.nan, None], nan_as_null=False),
        Series([1.1032, 2.32, 43.4], index=[0, 4, 3]),
        Series([]),
        Series([-3]),
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        np.random.normal(-100, 100, 1000),
        np.random.randint(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        Series([1.1, 2.32, 43.4], index=[0, 500, 4000]),
        Series([5]),
    ],
)
def test_corr1d(data1, data2):
    gs1 = Series(data1)
    gs2 = Series(data2)

    ps1 = gs1.to_pandas(nullable_pd_dtype=False)
    ps2 = gs2.to_pandas(nullable_pd_dtype=False)

    got = gs1.corr(gs2)
    expected = ps1.corr(ps2)
    np.testing.assert_approx_equal(got, expected, significant=8)


def test_df_corr():

    gdf = randomdata(100, {str(x): float for x in range(50)})
    pdf = gdf.to_pandas()
    got = gdf.corr()
    expected = pdf.corr()
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [0.0, 1, 3, 6, np.NaN, 7, 5.0, np.nan, 5, 2, 3, -100],
        [np.nan] * 3,
        [1, 5, 3],
        [],
    ],
)
@pytest.mark.parametrize(
    "ops",
    [
        "mean",
        "min",
        "max",
        "sum",
        "product",
        "var",
        "std",
        "prod",
        "kurtosis",
        "skew",
        "any",
        "all",
        "cummin",
        "cummax",
        "cumsum",
        "cumprod",
    ],
)
@pytest.mark.parametrize("skipna", [True, False, None])
def test_nans_stats(data, ops, skipna):
    psr = pd.Series(data)
    gsr = Series(data)
    assert_eq(
        getattr(psr, ops)(skipna=skipna), getattr(gsr, ops)(skipna=skipna)
    )

    psr = pd.Series(data)
    gsr = Series(data, nan_as_null=False)
    # Since there is no concept of `nan_as_null` in pandas,
    # nulls will be returned in the operations. So only
    # testing for `skipna=True` when `nan_as_null=False`
    assert_eq(getattr(psr, ops)(skipna=True), getattr(gsr, ops)(skipna=True))


@pytest.mark.parametrize(
    "data",
    [
        [0.0, 1, 3, 6, np.NaN, 7, 5.0, np.nan, 5, 2, 3, -100],
        [np.nan] * 3,
        [1, 5, 3],
    ],
)
@pytest.mark.parametrize("ops", ["sum", "product", "prod"])
@pytest.mark.parametrize("skipna", [True, False, None])
@pytest.mark.parametrize("min_count", [-10, -1, 0, 1, 2, 3, 5, 10])
def test_min_count_ops(data, ops, skipna, min_count):
    psr = pd.Series(data)
    gsr = Series(data)

    assert_eq(
        getattr(psr, ops)(skipna=skipna, min_count=min_count),
        getattr(gsr, ops)(skipna=skipna, min_count=min_count),
    )
