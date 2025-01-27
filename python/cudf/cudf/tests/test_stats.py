# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.datasets import randomdata
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if

params_dtypes = [np.int32, np.uint32, np.float32, np.float64]
methods = ["min", "max", "sum", "mean", "var", "std"]

interpolation_methods = ["linear", "lower", "higher", "midpoint", "nearest"]


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("dtype", params_dtypes)
@pytest.mark.parametrize("skipna", [True, False])
def test_series_reductions(method, dtype, skipna):
    rng = np.random.default_rng(seed=0)
    arr = rng.random(100)
    if np.issubdtype(dtype, np.integer):
        arr *= 100
        mask = arr > 10
    else:
        mask = arr > 0.5

    arr = arr.astype(dtype)
    if dtype in (np.float32, np.float64):
        arr[[2, 5, 14, 19, 50, 70]] = np.nan
    sr = cudf.Series(arr)
    sr[~mask] = None
    psr = sr.to_pandas()
    psr[~mask] = np.nan

    def call_test(sr, skipna):
        fn = getattr(sr, method)
        if method in ["std", "var"]:
            return fn(ddof=1, skipna=skipna)
        else:
            return fn(skipna=skipna)

    expect, got = call_test(psr, skipna=skipna), call_test(sr, skipna=skipna)

    np.testing.assert_approx_equal(expect, got)


@pytest.mark.parametrize("method", methods)
def test_series_reductions_concurrency(method):
    e = ThreadPoolExecutor(10)

    rng = np.random.default_rng(seed=0)
    srs = [cudf.Series(rng.random(10000)) for _ in range(1)]

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
    rng = np.random.default_rng(seed=0)
    arr = rng.random(100) - 0.5
    sr = cudf.Series(arr)
    pd = sr.to_pandas()
    got = sr.std(ddof=ddof)
    expect = pd.std(ddof=ddof)
    np.testing.assert_approx_equal(expect, got)


def test_series_unique():
    rng = np.random.default_rng(seed=0)
    for size in [10**x for x in range(5)]:
        arr = rng.integers(low=-1, high=10, size=size)
        mask = arr != -1
        sr = cudf.Series(arr)
        sr[~mask] = None
        assert set(arr[mask]) == set(sr.unique().dropna().to_numpy())
        assert len(set(arr[mask])) == sr.nunique()


@pytest.mark.parametrize(
    "nan_as_null, dropna",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_series_nunique(nan_as_null, dropna):
    # We remove nulls as opposed to NaNs using the dropna parameter,
    # so to test against pandas we replace NaN with another discrete value
    cudf_series = cudf.Series([1, 2, 2, 3, 3], nan_as_null=nan_as_null)
    pd_series = pd.Series([1, 2, 2, 3, 3])
    expect = pd_series.nunique(dropna=dropna)
    got = cudf_series.nunique(dropna=dropna)
    assert expect == got

    cudf_series = cudf.Series(
        [1.0, 2.0, 3.0, np.nan, None], nan_as_null=nan_as_null
    )
    if nan_as_null is True:
        pd_series = pd.Series([1.0, 2.0, 3.0, np.nan, None])
    else:
        pd_series = pd.Series([1.0, 2.0, 3.0, -1.0, None])

    expect = pd_series.nunique(dropna=dropna)
    got = cudf_series.nunique(dropna=dropna)
    assert expect == got

    cudf_series = cudf.Series([1.0, np.nan, np.nan], nan_as_null=nan_as_null)
    if nan_as_null is True:
        pd_series = pd.Series([1.0, np.nan, np.nan])
    else:
        pd_series = pd.Series([1.0, -1.0, -1.0])
    expect = pd_series.nunique(dropna=dropna)
    got = cudf_series.nunique(dropna=dropna)
    assert expect == got


def test_series_scale():
    rng = np.random.default_rng(seed=0)
    arr = pd.Series(rng.integers(low=-10, high=10, size=100))
    sr = cudf.Series(arr)

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
    gdf_series = cudf.Series(arr)

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
    gdf_series = cudf.Series(arr)

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

    gdf_series = cudf.Series(arr)
    pdf_series = pd.Series(arr)

    q1 = gdf_series.quantile(quant_values, exact=False)
    q2 = pdf_series.quantile(quant_values)

    assert_eq(q1, q2)


def test_approx_quantiles_int():
    arr = np.asarray([1, 2, 3])
    quant_values = [0.5]
    approx_results = [2]

    gdf_series = cudf.Series(arr)

    q1 = gdf_series.quantile(quant_values, exact=False)

    assert approx_results == q1.to_pandas().values


@pytest.mark.parametrize("data", [[], [1, 2, 3, 10, 326497]])
@pytest.mark.parametrize(
    "q",
    [
        [],
        0.5,
        1,
        0.234,
        [0.345],
        [0.243, 0.5, 1],
        np.array([0.5, 1]),
        cp.array([0.5, 1]),
    ],
)
def test_misc_quantiles(data, q):
    pdf_series = pd.Series(data, dtype="float64" if len(data) == 0 else None)
    gdf_series = cudf.from_pandas(pdf_series)

    expected = pdf_series.quantile(q.get() if isinstance(q, cp.ndarray) else q)
    actual = gdf_series.quantile(q)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        {"data": np.random.default_rng(seed=0).normal(-100, 100, 1000)},
        {"data": np.random.default_rng(seed=0).integers(-50, 50, 1000)},
        {"data": (np.zeros(100))},
        {"data": np.repeat(np.nan, 100)},
        {"data": np.array([1.123, 2.343, np.nan, 0.0])},
        {
            "data": [5, 10, 53, None, np.nan, None, 12, 43, -423],
            "nan_as_null": False,
        },
        {"data": [1.1032, 2.32, 43.4, 13, -312.0], "index": [0, 4, 3, 19, 6]},
        {"data": [], "dtype": "float64"},
        {"data": [-3]},
    ],
)
@pytest.mark.parametrize("null_flag", [False, True])
@pytest.mark.parametrize("numeric_only", [False, True])
def test_kurtosis_series(data, null_flag, numeric_only):
    gs = cudf.Series(**data)
    ps = gs.to_pandas()

    if null_flag and len(gs) > 2:
        gs.iloc[[0, 2]] = None
        ps.iloc[[0, 2]] = None

    got = gs.kurtosis(numeric_only=numeric_only)
    expected = ps.kurtosis(numeric_only=numeric_only)

    assert_eq(got, expected)

    got = gs.kurt(numeric_only=numeric_only)
    expected = ps.kurt(numeric_only=numeric_only)

    assert_eq(got, expected)


@pytest.mark.parametrize("op", ["skew", "kurt"])
def test_kurt_skew_error(op):
    gs = cudf.Series(["ab", "cd"])
    ps = gs.to_pandas()

    assert_exceptions_equal(
        getattr(gs, op),
        getattr(ps, op),
        lfunc_args_and_kwargs=([], {"numeric_only": True}),
        rfunc_args_and_kwargs=([], {"numeric_only": True}),
    )


@pytest.mark.parametrize(
    "data",
    [
        cudf.Series(np.random.default_rng(seed=0).normal(-100, 100, 1000)),
        cudf.Series(np.random.default_rng(seed=0).integers(-50, 50, 1000)),
        cudf.Series(np.zeros(100)),
        cudf.Series(np.repeat(np.nan, 100)),
        cudf.Series(np.array([1.123, 2.343, np.nan, 0.0])),
        cudf.Series(
            [5, 10, 53, None, np.nan, None, 12, 43, -423], nan_as_null=False
        ),
        cudf.Series([1.1032, 2.32, 43.4, 13, -312.0], index=[0, 4, 3, 19, 6]),
        cudf.Series([], dtype="float64"),
        cudf.Series([-3]),
    ],
)
@pytest.mark.parametrize("null_flag", [False, True])
@pytest.mark.parametrize("numeric_only", [False, True])
def test_skew_series(data, null_flag, numeric_only):
    pdata = data.to_pandas()

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.skew(numeric_only=numeric_only)
    expected = pdata.skew(numeric_only=numeric_only)

    assert_eq(got, expected)


@pytest.mark.parametrize("dtype", params_dtypes)
@pytest.mark.parametrize("num_na", [0, 1, 50, 99, 100])
def test_series_median(dtype, num_na):
    rng = np.random.default_rng(seed=0)
    arr = rng.random(100)
    if np.issubdtype(dtype, np.integer):
        arr *= 100
    mask = np.arange(100) >= num_na

    arr = arr.astype(dtype)
    sr = cudf.Series(arr)
    sr[~mask] = None
    arr2 = arr[mask]
    ps = pd.Series(arr2, dtype=dtype)

    actual = sr.median(skipna=True)
    desired = ps.median(skipna=True)

    np.testing.assert_approx_equal(actual, desired)

    # only for float until integer null supported convert to pandas in cudf
    # eg. pd.Int64Dtype
    if np.issubdtype(dtype, np.floating):
        ps = sr.to_pandas()
        actual = sr.median(skipna=False)
        desired = ps.median(skipna=False)
        np.testing.assert_approx_equal(actual, desired)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "data",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        np.array([-2, 3.75, 6, None, None, None, -8.5, None, 4.2]),
        cudf.Series([], dtype="float64"),
        cudf.Series([-3]),
    ],
)
@pytest.mark.parametrize("periods", range(-5, 5))
@pytest.mark.parametrize(
    "fill_method", ["ffill", "bfill", "pad", "backfill", no_default, None]
)
def test_series_pct_change(data, periods, fill_method):
    cs = cudf.Series(data)
    ps = cs.to_pandas()

    if np.abs(periods) <= len(cs):
        with expect_warning_if(fill_method not in (no_default, None)):
            got = cs.pct_change(periods=periods, fill_method=fill_method)
        with expect_warning_if(
            (
                fill_method not in (no_default, None)
                or (fill_method is not None and ps.isna().any())
            )
        ):
            expected = ps.pct_change(periods=periods, fill_method=fill_method)
        np.testing.assert_array_almost_equal(
            got.to_numpy(na_value=np.nan), expected
        )


@pytest.mark.parametrize(
    "data1",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        cudf.Series([5, 10, 53, None, np.nan, None], nan_as_null=False),
        cudf.Series([1.1, 2.32, 43.4], index=[0, 4, 3]),
        cudf.Series([], dtype="float64"),
        cudf.Series([-3]),
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        cudf.Series([1.1, 2.32, 43.4], index=[0, 500, 4000]),
        cudf.Series([5]),
    ],
)
def test_cov1d(data1, data2):
    gs1 = cudf.Series(data1)
    gs2 = cudf.Series(data2)

    ps1 = gs1.to_pandas()
    ps2 = gs2.to_pandas()

    got = gs1.cov(gs2)
    ps1_align, ps2_align = ps1.align(ps2, join="inner")
    with expect_warning_if(
        (len(ps1_align.dropna()) == 1 and len(ps2_align.dropna()) > 0)
        or (len(ps2_align.dropna()) == 1 and len(ps1_align.dropna()) > 0),
        RuntimeWarning,
    ):
        expected = ps1.cov(ps2)
    np.testing.assert_approx_equal(got, expected, significant=8)


@pytest.mark.parametrize(
    "data1",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        cudf.Series([5, 10, 53, None, np.nan, None], nan_as_null=False),
        cudf.Series([1.1032, 2.32, 43.4], index=[0, 4, 3]),
        cudf.Series([], dtype="float64"),
        cudf.Series([-3]),
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        cudf.Series([1.1, 2.32, 43.4], index=[0, 500, 4000]),
        cudf.Series([5]),
    ],
)
@pytest.mark.parametrize("method", ["spearman", "pearson"])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warnings missing on older pandas (scipy version seems unrelated?)",
)
def test_corr1d(data1, data2, method):
    if method == "spearman":
        # Pandas uses scipy.stats.spearmanr code-path
        pytest.importorskip("scipy")

    gs1 = cudf.Series(data1)
    gs2 = cudf.Series(data2)

    ps1 = gs1.to_pandas()
    ps2 = gs2.to_pandas()

    got = gs1.corr(gs2, method)

    ps1_align, ps2_align = ps1.align(ps2, join="inner")

    is_singular = (
        len(ps1_align.dropna()) == 1 and len(ps2_align.dropna()) > 0
    ) or (len(ps2_align.dropna()) == 1 and len(ps1_align.dropna()) > 0)
    is_identical = (
        len(ps1_align.dropna().unique()) == 1 and len(ps2_align.dropna()) > 0
    ) or (
        len(ps2_align.dropna().unique()) == 1 and len(ps1_align.dropna()) > 0
    )

    # Pearson correlation leads to division by 0 when either sample size is 1.
    # Spearman allows for size 1 samples, but will error if all data in a
    # sample is identical since the covariance is zero and so the correlation
    # coefficient is not defined.
    cond = ((is_singular or is_identical) and method == "pearson") or (
        is_identical and not is_singular and method == "spearman"
    )
    if method == "spearman":
        # SciPy has shuffled around the warning it throws a couple of times.
        # It's not worth the effort of conditionally importing the appropriate
        # warning based on the scipy version, just catching a base Warning is
        # good enough validation.
        expected_warning = Warning
    elif method == "pearson":
        expected_warning = RuntimeWarning

    with expect_warning_if(cond, expected_warning):
        expected = ps1.corr(ps2, method)
    np.testing.assert_approx_equal(got, expected, significant=8)


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_df_corr(method):
    gdf = randomdata(100, {str(x): float for x in range(50)})
    pdf = gdf.to_pandas()
    got = gdf.corr(method)
    expected = pdf.corr(method)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [0.0, 1, 3, 6, np.nan, 7, 5.0, np.nan, 5, 2, 3, -100],
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
@pytest.mark.parametrize("skipna", [True, False])
def test_nans_stats(data, ops, skipna):
    psr = pd.Series(data, dtype="float64" if len(data) == 0 else None)
    gsr = cudf.from_pandas(psr)

    assert_eq(
        getattr(psr, ops)(skipna=skipna), getattr(gsr, ops)(skipna=skipna)
    )

    gsr = cudf.Series(
        data, dtype="float64" if len(data) == 0 else None, nan_as_null=False
    )
    # Since there is no concept of `nan_as_null` in pandas,
    # nulls will be returned in the operations. So only
    # testing for `skipna=True` when `nan_as_null=False`
    assert_eq(getattr(psr, ops)(skipna=True), getattr(gsr, ops)(skipna=True))


@pytest.mark.parametrize(
    "data",
    [
        [0.0, 1, 3, 6, np.nan, 7, 5.0, np.nan, 5, 2, 3, -100],
        [np.nan] * 3,
        [1, 5, 3],
    ],
)
@pytest.mark.parametrize("ops", ["sum", "product", "prod"])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("min_count", [-10, -1, 0, 1, 2, 3, 5, 10])
def test_min_count_ops(data, ops, skipna, min_count):
    psr = pd.Series(data)
    gsr = cudf.Series(data, nan_as_null=False)

    assert_eq(
        getattr(psr, ops)(skipna=skipna, min_count=min_count),
        getattr(gsr, ops)(skipna=skipna, min_count=min_count),
    )


@pytest.mark.parametrize(
    "data1",
    [
        [1, 2, 3, 4],
        [10, 1, 3, 5],
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        [1, 2, 3, 4],
        [10, 1, 3, 5],
    ],
)
@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_cov_corr_datetime_timedelta(data1, data2, dtype):
    gsr1 = cudf.Series(data1, dtype=dtype)
    gsr2 = cudf.Series(data2, dtype=dtype)
    psr1 = gsr1.to_pandas()
    psr2 = gsr2.to_pandas()

    assert_eq(psr1.corr(psr2), gsr1.corr(gsr2))
    assert_eq(psr1.cov(psr2), gsr1.cov(gsr2))


@pytest.mark.parametrize(
    "data",
    [
        randomdata(
            nrows=1000, dtypes={"a": float, "b": int, "c": float, "d": str}
        ),
    ],
)
@pytest.mark.parametrize("null_flag", [False, True])
@pytest.mark.parametrize("numeric_only", [False, True])
def test_kurtosis_df(data, null_flag, numeric_only):
    if not numeric_only:
        data = data.select_dtypes(include="number")
    pdata = data.to_pandas()

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.kurtosis(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()

    expected = pdata.kurtosis(numeric_only=numeric_only)
    np.testing.assert_array_almost_equal(got, expected)

    got = data.kurt(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()

    expected = pdata.kurt(numeric_only=numeric_only)
    np.testing.assert_array_almost_equal(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        randomdata(
            nrows=1000, dtypes={"a": float, "b": int, "c": float, "d": str}
        ),
    ],
)
@pytest.mark.parametrize("null_flag", [False, True])
@pytest.mark.parametrize("numeric_only", [False, True])
def test_skew_df(data, null_flag, numeric_only):
    if not numeric_only:
        data = data.select_dtypes(include="number")
    pdata = data.to_pandas()

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.skew(numeric_only=numeric_only)
    expected = pdata.skew(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()
    np.testing.assert_array_almost_equal(got, expected)
