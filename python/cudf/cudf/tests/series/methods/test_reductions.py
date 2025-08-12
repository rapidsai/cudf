# Copyright (c) 2019-2025, NVIDIA CORPORATION.
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("data", [[], [1, 2, 3]])
def test_series_pandas_methods(data, reduction_methods):
    arr = np.array(data)
    sr = cudf.Series(arr)
    psr = pd.Series(arr)
    np.testing.assert_equal(
        getattr(sr, reduction_methods)(), getattr(psr, reduction_methods)()
    )


def test_series_reductions(
    request, reduction_methods, numeric_types_as_str, skipna
):
    request.applymarker(
        pytest.mark.xfail(
            reduction_methods == "quantile",
            raises=TypeError,
            reason=f"{reduction_methods} doesn't support skipna",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            reduction_methods == "product"
            and skipna
            and numeric_types_as_str not in {"float32", "float64"},
            reason=f"{reduction_methods} incorrect with {skipna=}",
        )
    )
    rng = np.random.default_rng(seed=0)
    arr = rng.random(100)
    if np.dtype(numeric_types_as_str).kind in "iu":
        arr *= 100
        mask = arr > 10
    else:
        mask = arr > 0.5

    arr = arr.astype(numeric_types_as_str)
    if numeric_types_as_str in ("float32", "float64"):
        arr[[2, 5, 14, 19, 50, 70]] = np.nan
    sr = cudf.Series(arr)
    sr[~mask] = None
    psr = sr.to_pandas()
    psr[~mask] = np.nan

    def call_test(sr, skipna):
        fn = getattr(sr, reduction_methods)
        if reduction_methods in ["std", "var"]:
            return fn(ddof=1, skipna=skipna)
        else:
            return fn(skipna=skipna)

    expect = call_test(psr, skipna=skipna)
    got = call_test(sr, skipna=skipna)

    np.testing.assert_approx_equal(expect, got, significant=4)


def test_series_reductions_concurrency(reduction_methods):
    rng = np.random.default_rng(seed=0)
    srs = [cudf.Series(rng.random(100))]

    def call_test(sr):
        fn = getattr(sr, reduction_methods)
        if reduction_methods in ["std", "var"]:
            return fn(ddof=1)
        else:
            return fn()

    def f(sr):
        return call_test(sr + 1)

    with ThreadPoolExecutor(10) as e:
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


def test_exact_quantiles(quantile_interpolation):
    arr = np.asarray([6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7])
    quant_values = [0.0, 0.25, 0.33, 0.5, 1.0]

    df = pd.DataFrame(arr)
    gdf_series = cudf.Series(arr)

    q1 = gdf_series.quantile(
        quant_values, interpolation=quantile_interpolation, exact=True
    )

    q2 = df.quantile(quant_values, interpolation=quantile_interpolation)

    np.testing.assert_allclose(
        q1.to_pandas().values, np.array(q2.values).T.flatten(), rtol=1e-10
    )


def test_exact_quantiles_int(quantile_interpolation):
    arr = np.asarray([7, 0, 3, 4, 2, 1, -1, 1, 6])
    quant_values = [0.0, 0.25, 0.33, 0.5, 1.0]

    df = pd.DataFrame(arr)
    gdf_series = cudf.Series(arr)

    q1 = gdf_series.quantile(
        quant_values, interpolation=quantile_interpolation, exact=True
    )

    q2 = df.quantile(quant_values, interpolation=quantile_interpolation)

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
    "data, index, dtype, nan_as_null",
    [
        [
            np.random.default_rng(seed=0).normal(-100, 100, 1000),
            None,
            None,
            None,
        ],
        [
            np.random.default_rng(seed=0).integers(-50, 50, 1000),
            None,
            None,
            None,
        ],
        [np.zeros(100), None, None, None],
        [np.repeat(np.nan, 100), None, None, None],
        [np.array([1.123, 2.343, np.nan, 0.0]), None, None, None],
        [[5, 10, 53, None, np.nan, None, 12, 43, -423], None, None, False],
        [[1.1032, 2.32, 43.4, 13, -312.0], [0, 4, 3, 19, 6], None, None],
        [[], None, "float64", None],
        [[-3], None, None, None],
    ],
)
@pytest.mark.parametrize("null_flag", [False, True])
def test_skew_series(data, index, dtype, nan_as_null, null_flag, numeric_only):
    data = cudf.Series(data, index=index, dtype=dtype, nan_as_null=nan_as_null)
    pdata = data.to_pandas()

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.skew(numeric_only=numeric_only)
    expected = pdata.skew(numeric_only=numeric_only)

    assert_eq(got, expected)


@pytest.mark.parametrize("num_na", [0, 50, 100])
def test_series_median(numeric_types_as_str, num_na):
    rng = np.random.default_rng(seed=0)
    arr = rng.random(100)
    dtype = np.dtype(numeric_types_as_str)
    if dtype.kind in "iu":
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
    if dtype.kind == "f":
        ps = sr.to_pandas()
        actual = sr.median(skipna=False)
        desired = ps.median(skipna=False)
        np.testing.assert_approx_equal(actual, desired)


@pytest.mark.parametrize(
    "data",
    [
        [0.0, 1, 3, 6, np.nan, 7, 5.0, np.nan, 5, 2, 3, -100],
        [np.nan] * 3,
        [1, 5, 3],
        [],
    ],
)
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_nans_stats(data, reduction_methods, skipna, request):
    request.applymarker(
        pytest.mark.xfail(
            reduction_methods == "quantile",
            raises=TypeError,
            reason=f"{reduction_methods} doesn't support skipna",
        )
    )

    psr = pd.Series(data, dtype="float64" if len(data) == 0 else None)
    gsr = cudf.from_pandas(psr)

    assert_eq(
        getattr(psr, reduction_methods)(skipna=skipna),
        getattr(gsr, reduction_methods)(skipna=skipna),
    )

    gsr = cudf.Series(
        data, dtype="float64" if len(data) == 0 else None, nan_as_null=False
    )
    # Since there is no concept of `nan_as_null` in pandas,
    # nulls will be returned in the operations. So only
    # testing for `skipna=True` when `nan_as_null=False`
    assert_eq(
        getattr(psr, reduction_methods)(skipna=True),
        getattr(gsr, reduction_methods)(skipna=True),
    )


@pytest.mark.parametrize(
    "data",
    [
        [0.0, 1, 3, 6, np.nan, 7, 5.0, np.nan, 5, 2, 3, -100],
        [np.nan] * 3,
        [1, 5, 3],
    ],
)
@pytest.mark.parametrize("min_count", [-10, -1, 0, 1, 5, 10])
def test_min_count_ops(data, request, reduction_methods, skipna, min_count):
    request.applymarker(
        pytest.mark.xfail(
            reduction_methods == "quantile",
            raises=TypeError,
            reason=f"{reduction_methods} doesn't support skipna",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            reduction_methods
            in {
                "skew",
                "kurtosis",
                "median",
                "var",
                "std",
                "any",
                "all",
                "max",
                "min",
            },
            raises=TypeError,
            reason=f"{reduction_methods} doesn't support min_count",
        )
    )
    psr = pd.Series(data)
    gsr = cudf.Series(data, nan_as_null=False)

    assert_eq(
        getattr(psr, reduction_methods)(skipna=skipna, min_count=min_count),
        getattr(gsr, reduction_methods)(skipna=skipna, min_count=min_count),
    )
