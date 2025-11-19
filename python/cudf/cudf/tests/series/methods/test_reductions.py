# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import re
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from packaging.version import parse

import cudf
from cudf.core._compat import PANDAS_GE_230
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


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


@pytest.mark.parametrize("q", [2, [1, 2, 3]])
def test_quantile_range_error(q):
    ps = pd.Series([1, 2, 3])
    gs = cudf.from_pandas(ps)
    assert_exceptions_equal(
        lfunc=ps.quantile,
        rfunc=gs.quantile,
        lfunc_args_and_kwargs=([q],),
        rfunc_args_and_kwargs=([q],),
    )


def test_quantile_q_type():
    gs = cudf.Series([1, 2, 3])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "q must be a scalar or array-like, got <class "
            "'cudf.core.dataframe.DataFrame'>"
        ),
    ):
        gs.quantile(cudf.DataFrame())


def test_quantile_type_int_float(quantile_interpolation):
    data = [1, 3, 4]
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    expected = psr.quantile(0.5, interpolation=quantile_interpolation)
    actual = gsr.quantile(0.5, interpolation=quantile_interpolation)

    assert expected == actual
    assert type(expected) is type(actual)


@pytest.mark.parametrize("val", [0.9, float("nan")])
def test_ignore_nans(val):
    data = [float("nan"), float("nan"), val]
    psr = pd.Series(data)
    gsr = cudf.Series(data, nan_as_null=False)

    expected = gsr.quantile(0.9)
    result = psr.quantile(0.9)
    assert_eq(result, expected)


def test_sum(numeric_types_as_str):
    data = np.arange(10, dtype=numeric_types_as_str)
    sr = cudf.Series(data)

    got = sr.sum()
    expect = data.sum()
    significant = 4 if numeric_types_as_str == "float32" else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


@pytest.mark.parametrize(
    "middle, expected",
    [
        ("there", "HellothereWorld"),
        (None, "HelloWorld"),
    ],
)
def test_sum_string(middle, expected):
    s = cudf.Series(["Hello", middle, "World"])

    assert s.sum() == expected


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            cudf.Decimal64Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype(10, 6),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype(16, 7),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal32Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal32 format string only supported in pyarrow >=19",
            ),
        ),
        cudf.Decimal128Dtype(20, 7),
    ],
)
def test_sum_decimal(dtype):
    data = [str(x) for x in np.array([1, 11, 111]) / 100]

    expected = pd.Series([Decimal(x) for x in data]).sum()
    got = cudf.Series(data).astype(dtype).sum()

    assert_eq(expected, got)


def test_product(numeric_types_as_str):
    data = np.arange(10, dtype=numeric_types_as_str)
    sr = cudf.Series(data)

    got = sr.product()
    expect = pd.Series(data).product()
    significant = 4 if numeric_types_as_str == "float32" else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            cudf.Decimal64Dtype(6, 2),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype(8, 4),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype(10, 5),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal32Dtype(6, 2),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal32 format string only supported in pyarrow >=19",
            ),
        ),
        cudf.Decimal128Dtype(20, 5),
    ],
)
def test_product_decimal(dtype):
    data = [str(x) for x in np.array([1, 11, 111]) / 100]

    expected = pd.Series([Decimal(x) for x in data]).product()
    got = cudf.Series(data).astype(dtype).product()

    assert_eq(expected, got)


def test_sum_of_squares(numeric_types_as_str):
    accuracy_for_dtype = {"float64": 6, "float32": 5}
    data = np.arange(10, dtype=numeric_types_as_str)
    sr = cudf.Series(data)
    df = cudf.DataFrame(sr)

    got = (sr**2).sum()
    got_df = (df**2).sum()
    expect = (data**2).sum()

    if "int" in numeric_types_as_str:
        np.testing.assert_array_almost_equal(expect, got)
        np.testing.assert_array_almost_equal(expect, got_df.iloc[0])
    else:
        np.testing.assert_approx_equal(
            expect, got, significant=accuracy_for_dtype[numeric_types_as_str]
        )
        np.testing.assert_approx_equal(
            expect,
            got_df.iloc[0],
            significant=accuracy_for_dtype[numeric_types_as_str],
        )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            cudf.Decimal64Dtype(6, 2),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype(8, 4),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype(10, 5),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        cudf.Decimal128Dtype(20, 7),
        pytest.param(
            cudf.Decimal32Dtype(6, 2),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal32 format string only supported in pyarrow >=19",
            ),
        ),
    ],
)
def test_sum_of_squares_decimal(dtype):
    data = [str(x) for x in np.array([1, 11, 111]) / 100]

    expected = pd.Series([Decimal(x) for x in data]).pow(2).sum()
    got = (cudf.Series(data).astype(dtype) ** 2).sum()

    assert_eq(expected, got)


def test_min(numeric_types_as_str):
    data = np.arange(10, dtype=numeric_types_as_str)
    sr = cudf.Series(data)

    got = sr.min()
    expect = getattr(np, numeric_types_as_str)(data.min())

    assert expect == got


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            cudf.Decimal64Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype(10, 6),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype(16, 7),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal32Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal32 format string only supported in pyarrow >=19",
            ),
        ),
        cudf.Decimal128Dtype(20, 7),
    ],
)
def test_min_decimal(dtype):
    data = [str(x) for x in np.array([1, 11, 111]) / 100]

    expected = pd.Series([Decimal(x) for x in data]).min()
    got = cudf.Series(data).astype(dtype).min()

    assert_eq(expected, got)


def test_max(numeric_types_as_str):
    data = np.arange(10, dtype=numeric_types_as_str)
    sr = cudf.Series(data)

    got = sr.max()
    expect = getattr(np, numeric_types_as_str)(data.max())

    assert expect == got


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            cudf.Decimal64Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype(10, 6),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype(16, 7),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            cudf.Decimal32Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0.0"),
                reason="decimal32 format string only supported in pyarrow >=19",
            ),
        ),
        cudf.Decimal128Dtype(20, 7),
    ],
)
def test_max_decimal(dtype):
    data = [str(x) for x in np.array([1, 11, 111]) / 100]

    expected = pd.Series([Decimal(x) for x in data]).max()
    got = cudf.Series(data).astype(dtype).max()

    assert_eq(expected, got)


def test_sum_masked():
    data = np.array([1.1, 1.2, np.nan], dtype="float64")
    sr = cudf.Series(data, nan_as_null=True)

    got = sr.sum()
    expected = np.nansum(data)
    np.testing.assert_approx_equal(expected, got)


def test_sum_boolean():
    s = cudf.Series(np.arange(100000))
    got = (s > 1).sum()
    expect = 99998

    assert expect == got


def test_date_minmax():
    rng = np.random.default_rng(seed=0)
    np_data = rng.normal(size=10)
    gdf_data = cudf.Series(np_data)

    np_casted = np_data.astype("datetime64[ms]")
    gdf_casted = gdf_data.astype("datetime64[ms]")

    np_min = np_casted.min()
    gdf_min = gdf_casted.min()
    assert np_min == gdf_min

    np_max = np_casted.max()
    gdf_max = gdf_casted.max()
    assert np_max == gdf_max


@pytest.mark.parametrize(
    "op",
    ["sum", "product", "var", "kurt", "kurtosis", "skew"],
)
def test_datetime_unsupported_reductions(op):
    gsr = cudf.Series([1, 2, 3, None], dtype="datetime64[ns]")
    psr = gsr.to_pandas()

    assert_exceptions_equal(
        lfunc=getattr(psr, op),
        rfunc=getattr(gsr, op),
    )


@pytest.mark.parametrize("op", ["product", "var", "kurt", "kurtosis", "skew"])
def test_timedelta_unsupported_reductions(op):
    gsr = cudf.Series([1, 2, 3, None], dtype="timedelta64[ns]")
    psr = gsr.to_pandas()

    assert_exceptions_equal(
        lfunc=getattr(psr, op),
        rfunc=getattr(gsr, op),
    )


def test_categorical_reductions(request, reduction_methods):
    request.applymarker(
        pytest.mark.xfail(
            reduction_methods in ["quantile", "all", "any"],
            reason=f"{reduction_methods} didn't fail",
        )
    )

    gsr = cudf.Series([1, 2, 3, None], dtype="category")
    psr = gsr.to_pandas()

    assert_exceptions_equal(
        getattr(psr, reduction_methods), getattr(gsr, reduction_methods)
    )


@pytest.mark.parametrize(
    "data_non_overflow",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
def test_timedelta_reduction_ops(
    data_non_overflow, timedelta_types_as_str, reduction_methods
):
    if reduction_methods not in ["sum", "mean", "median", "quantile"]:
        pytest.skip(f"{reduction_methods} not supported for timedelta")
    gsr = cudf.Series(data_non_overflow, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    if len(psr) > 0 and psr.isnull().all() and reduction_methods == "median":
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            expected = getattr(psr, reduction_methods)()
    else:
        with expect_warning_if(
            PANDAS_GE_230
            and reduction_methods == "quantile"
            and len(data_non_overflow) == 0
            and timedelta_types_as_str != "timedelta64[ns]"
        ):
            expected = getattr(psr, reduction_methods)()
    actual = getattr(gsr, reduction_methods)()
    if pd.isna(expected) and pd.isna(actual):
        pass
    elif isinstance(expected, pd.Timedelta) and isinstance(
        actual, pd.Timedelta
    ):
        assert (
            expected.round(gsr._column.time_unit).value
            == actual.round(gsr._column.time_unit).value
        )
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize("data", [[1, 2, 3], [], [1, 20, 1000, None]])
@pytest.mark.parametrize("ddof", [1, 2, 3])
def test_timedelta_std(data, timedelta_types_as_str, ddof):
    gsr = cudf.Series(data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    expected = psr.std(ddof=ddof)
    actual = gsr.std(ddof=ddof)

    if np.isnat(expected.to_numpy()) and np.isnat(actual.to_numpy()):
        assert True
    else:
        np.testing.assert_allclose(
            expected.to_numpy().astype("float64"),
            actual.to_numpy().astype("float64"),
            rtol=1e-5,
            atol=0,
        )


@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 100],
        [10, None, 100, None, None],
        [None, None, None],
        [1231],
    ],
)
def test_timedelta_reductions(data, op, timedelta_types_as_str):
    sr = cudf.Series(data, dtype=timedelta_types_as_str)
    psr = sr.to_pandas()

    actual = getattr(sr, op)()
    expected = getattr(psr, op)()

    if np.isnat(expected.to_numpy()) and np.isnat(actual):
        assert True
    else:
        assert_eq(expected.to_numpy(), actual)


@pytest.mark.parametrize(
    "data",
    [
        ["a", "b", "c", "d", "e"],
        ["a", "z", ".", '"', "aa", "zz"],
        ["aa", "zz"],
        ["z", "a", "zz", "aa"],
        ["1", "2", "3", "4", "5"],
        [""],
        ["a"],
        ["hello"],
        ["small text", "this is a larger text......"],
        ["👋🏻", "🔥", "🥇"],
        ["This is 💯", "here is a calendar", "📅"],
        ["", ".", ";", "[", "]"],
        ["\t", ".", "\n", "\n\t", "\t\n"],
    ],
)
@pytest.mark.parametrize("op", ["min", "max", "sum"])
def test_str_reductions_supported(data, op):
    psr = pd.Series(data)
    sr = cudf.Series(data)

    assert_eq(getattr(psr, op)(), getattr(sr, op)())


def test_str_mean():
    sr = cudf.Series(["a", "b", "c", "d", "e"])

    with pytest.raises(TypeError):
        sr.mean()


def test_string_product():
    psr = pd.Series(["1", "2", "3", "4", "5"])
    sr = cudf.Series(["1", "2", "3", "4", "5"])

    assert_exceptions_equal(
        lfunc=psr.product,
        rfunc=sr.product,
    )


def test_string_var():
    psr = pd.Series(["1", "2", "3", "4", "5"])
    sr = cudf.Series(["1", "2", "3", "4", "5"])

    assert_exceptions_equal(lfunc=psr.var, rfunc=sr.var)


def test_string_std():
    psr = pd.Series(["1", "2", "3", "4", "5"])
    sr = cudf.Series(["1", "2", "3", "4", "5"])

    assert_exceptions_equal(lfunc=psr.std, rfunc=sr.std)


def test_string_reduction_error():
    s = cudf.Series([None, None], dtype="str")
    ps = s.to_pandas(nullable=True)
    assert_exceptions_equal(
        s.any,
        ps.any,
        lfunc_args_and_kwargs=([], {"skipna": False}),
        rfunc_args_and_kwargs=([], {"skipna": False}),
    )

    assert_exceptions_equal(
        s.all,
        ps.all,
        lfunc_args_and_kwargs=([], {"skipna": False}),
        rfunc_args_and_kwargs=([], {"skipna": False}),
    )


@pytest.mark.parametrize("data", [[1, 2, 3], [], [1, 20, 1000, None]])
def test_datetime_stats(data, datetime_types_as_str, reduction_methods):
    if reduction_methods not in ["mean", "quantile"]:
        pytest.skip(f"{reduction_methods} not applicable for test")
    gsr = cudf.Series(data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    with expect_warning_if(
        PANDAS_GE_230
        and reduction_methods == "quantile"
        and len(data) == 0
        and datetime_types_as_str != "datetime64[ns]"
    ):
        expected = getattr(psr, reduction_methods)()
    actual = getattr(gsr, reduction_methods)()

    if len(data) == 0:
        assert np.isnat(expected.to_numpy()) and np.isnat(actual.to_numpy())
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 100],
        [10, None, 100, None, None],
        [None, None, None],
        [1231],
    ],
)
def test_datetime_reductions(data, reduction_methods, datetime_types_as_str):
    if reduction_methods not in ["max", "min", "std", "median"]:
        pytest.skip(f"{reduction_methods} not applicable for test")
    sr = cudf.Series(data, dtype=datetime_types_as_str)
    psr = sr.to_pandas()

    actual = getattr(sr, reduction_methods)()
    with expect_warning_if(
        psr.size > 0 and psr.isnull().all() and reduction_methods == "median",
        RuntimeWarning,
    ):
        expected = getattr(psr, reduction_methods)()

    if (
        expected is pd.NaT
        and actual is pd.NaT
        or (np.isnat(expected.to_numpy()) and np.isnat(actual))
    ):
        assert True
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize("op", ["min", "max"])
def test_categorical_maxima(op):
    ser = cudf.Series(
        ["a", "d", "c", "z", "g"],
        dtype=cudf.CategoricalDtype(["z", "c", "g", "d", "a"], ordered=False),
    )
    assert not ser.cat.ordered

    # Cannot get extrema of unordered Categorical column
    with pytest.raises(TypeError, match="Categorical is not ordered"):
        getattr(ser, op)()

    # Max/min should work after converting to "ordered"
    ser_pd = ser.to_pandas()
    result = getattr(ser.cat.as_ordered(), op)()
    result_pd = getattr(ser_pd.cat.as_ordered(), op)()
    assert_eq(result, result_pd)
