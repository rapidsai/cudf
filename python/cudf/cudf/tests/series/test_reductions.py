# Copyright (c) 2019-2025, NVIDIA CORPORATION.

import re
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from packaging.version import parse

import cudf
from cudf import Series
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_230,
    PANDAS_VERSION,
)
from cudf.core.column.column import as_column
from cudf.core.dtypes import Decimal32Dtype, Decimal64Dtype, Decimal128Dtype
from cudf.testing import _utils as utils, assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
    expect_warning_if,
    gen_rand,
)


@pytest.mark.parametrize("data", [[], [1, 2, 3]])
def test_series_pandas_methods(data, reduction_methods):
    arr = np.array(data)
    sr = cudf.Series(arr)
    psr = pd.Series(arr)
    np.testing.assert_equal(
        getattr(sr, reduction_methods)(), getattr(psr, reduction_methods)()
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


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
)
def test_quantile_type_int_float(interpolation):
    data = [1, 3, 4]
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    expected = psr.quantile(0.5, interpolation=interpolation)
    actual = gsr.quantile(0.5, interpolation=interpolation)

    assert expected == actual
    assert type(expected) is type(actual)


@pytest.mark.parametrize("val", [0.9, float("nan")])
def test_quantile_ignore_nans(val):
    data = [float("nan"), float("nan"), val]
    psr = pd.Series(data)
    gsr = cudf.Series(data, nan_as_null=False)

    expected = gsr.quantile(0.9)
    result = psr.quantile(0.9)
    assert_eq(result, expected)


def test_sum(numeric_types_as_str):
    data = gen_rand(numeric_types_as_str, 5)
    sr = Series(data)

    got = sr.sum()
    expect = data.sum()
    significant = 4 if numeric_types_as_str == "float32" else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


def test_sum_string():
    s = Series(["Hello", "there", "World"])

    got = s.sum()
    expected = "HellothereWorld"

    assert got == expected

    s = Series(["Hello", None, "World"])

    got = s.sum()
    expected = "HelloWorld"

    assert got == expected


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            Decimal64Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal64Dtype(10, 6),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal64Dtype(16, 7),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal32Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal32 format string only supported in pyarrow >=19",
            ),
        ),
        Decimal128Dtype(20, 7),
    ],
)
def test_sum_decimal(dtype):
    data = [str(x) for x in gen_rand("int64", 5, seed=0) / 100]

    expected = pd.Series([Decimal(x) for x in data]).sum()
    got = cudf.Series(data).astype(dtype).sum()

    assert_eq(expected, got)


def test_product(numeric_types_as_str):
    rng = np.random.default_rng(seed=0)
    dtype = np.dtype(numeric_types_as_str)
    if dtype.kind in {"u", "i"}:
        data = np.ones(5, dtype=dtype)
        # Set at most 30 items to [0..2) to keep the value within 2^32
        for _ in range(30):
            data[rng.integers(low=0, high=5, size=1)] = rng.uniform() * 2
    else:
        data = gen_rand(dtype, 5)

    sr = Series(data)

    got = sr.product()
    expect = pd.Series(data).product()
    significant = 4 if dtype.type == np.float32 else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            Decimal64Dtype(6, 2),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal64Dtype(8, 4),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal64Dtype(10, 5),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal32Dtype(6, 2),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal32 format string only supported in pyarrow >=19",
            ),
        ),
        Decimal128Dtype(20, 5),
    ],
)
def test_product_decimal(dtype):
    data = [str(x) for x in gen_rand("int8", 3) / 10]

    expected = pd.Series([Decimal(x) for x in data]).product()
    got = cudf.Series(data).astype(dtype).product()

    assert_eq(expected, got)


def test_sum_of_squares(numeric_types_as_str):
    dtype = np.dtype(numeric_types_as_str)
    data = gen_rand(dtype, 5)
    sr = Series(data)
    df = cudf.DataFrame(sr)

    got = (sr**2).sum()
    got_df = (df**2).sum()
    expect = (data**2).sum()
    accuracy_for_dtype = {np.float64: 6, np.float32: 5}
    if dtype.kind in {"u", "i"}:
        np.testing.assert_array_almost_equal(expect, got)
        np.testing.assert_array_almost_equal(expect, got_df.iloc[0])
    else:
        np.testing.assert_approx_equal(
            expect, got, significant=accuracy_for_dtype[dtype.type]
        )
        np.testing.assert_approx_equal(
            expect, got_df.iloc[0], significant=accuracy_for_dtype[dtype.type]
        )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            Decimal64Dtype(6, 2),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal64Dtype(8, 4),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal64Dtype(10, 5),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        Decimal128Dtype(20, 7),
        pytest.param(
            Decimal32Dtype(6, 2),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal32 format string only supported in pyarrow >=19",
            ),
        ),
    ],
)
def test_sum_of_squares_decimal(dtype):
    data = [str(x) for x in gen_rand("int8", 3) / 10]

    expected = pd.Series([Decimal(x) for x in data]).pow(2).sum()
    got = (cudf.Series(data).astype(dtype) ** 2).sum()

    assert_eq(expected, got)


def test_min(numeric_types_as_str):
    dtype = np.dtype(numeric_types_as_str).type
    data = gen_rand(dtype, 5)
    sr = Series(data)

    got = sr.min()
    expect = dtype(data.min())

    assert expect == got


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            Decimal64Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal64Dtype(10, 6),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal64Dtype(16, 7),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal32Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal32 format string only supported in pyarrow >=19",
            ),
        ),
        Decimal128Dtype(20, 7),
    ],
)
def test_min_decimal(dtype):
    data = [str(x) for x in gen_rand("int64", 5) / 100]

    expected = pd.Series([Decimal(x) for x in data]).min()
    got = cudf.Series(data).astype(dtype).min()

    assert_eq(expected, got)


def test_max(numeric_types_as_str):
    dtype = np.dtype(numeric_types_as_str).type
    data = gen_rand(dtype, 5)
    sr = Series(data)

    got = sr.max()
    expect = dtype(data.max())

    assert expect == got


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            Decimal64Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal64Dtype(10, 6),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal64Dtype(16, 7),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal64 format string only supported in pyarrow >=19",
            ),
        ),
        pytest.param(
            Decimal32Dtype(6, 3),
            marks=pytest.mark.skipif(
                parse(pa.__version__) < parse("19.0"),
                reason="decimal32 format string only supported in pyarrow >=19",
            ),
        ),
        Decimal128Dtype(20, 7),
    ],
)
def test_max_decimal(dtype):
    data = [str(x) for x in gen_rand("int64", 5) / 100]

    expected = pd.Series([Decimal(x) for x in data]).max()
    got = cudf.Series(data).astype(dtype).max()

    assert_eq(expected, got)


def test_sum_masked():
    nelem = 5
    dtype = np.float64
    data = gen_rand(dtype, nelem)

    mask = utils.random_bitmask(nelem)
    bitmask = utils.expand_bits_to_bytes(mask)[:nelem]

    sr = Series._from_column(as_column(data).set_mask(mask))

    got = sr.sum()
    res_mask = np.asarray(bitmask, dtype=np.bool_)[: data.size]
    expect = data[res_mask].sum()

    significant = 4 if dtype == np.float32 else 6
    np.testing.assert_approx_equal(expect, got, significant=significant)


def test_sum_boolean():
    s = Series(np.arange(100000))
    got = (s > 1).sum()
    expect = 99998

    assert expect == got


def test_date_minmax():
    rng = np.random.default_rng(seed=0)
    np_data = rng.normal(size=10**3)
    gdf_data = Series(np_data)

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

    utils.assert_exceptions_equal(
        lfunc=getattr(psr, op),
        rfunc=getattr(gsr, op),
    )


@pytest.mark.parametrize("op", ["product", "var", "kurt", "kurtosis", "skew"])
def test_timedelta_unsupported_reductions(op):
    gsr = cudf.Series([1, 2, 3, None], dtype="timedelta64[ns]")
    psr = gsr.to_pandas()

    utils.assert_exceptions_equal(
        lfunc=getattr(psr, op),
        rfunc=getattr(gsr, op),
    )


@pytest.mark.parametrize("op", ["sum", "product", "std", "var"])
def test_categorical_unsupported_reductions(op):
    gsr = cudf.Series([1, 2, 3, None], dtype="category")
    psr = gsr.to_pandas()

    utils.assert_exceptions_equal(getattr(psr, op), getattr(gsr, op))


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warning not given on older versions of pandas",
)
def test_reductions_axis_none_warning(reduction_methods):
    if reduction_methods == "quantile":
        pytest.skip(f"pandas doesn't support {reduction_methods}")
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [10, 2, 3]})
    pdf = df.to_pandas()
    with expect_warning_if(
        reduction_methods in {"sum", "product", "std", "var"},
        FutureWarning,
    ):
        actual = getattr(df, reduction_methods)(axis=None)
    with expect_warning_if(
        reduction_methods in {"sum", "product", "std", "var"},
        FutureWarning,
    ):
        expected = getattr(pdf, reduction_methods)(axis=None)
    assert_eq(expected, actual, check_dtype=False)


def test_dataframe_reduction_no_args(reduction_methods):
    df = cudf.DataFrame({"a": range(10), "b": range(10)})
    pdf = df.to_pandas()
    result = getattr(df, reduction_methods)()
    expected = getattr(pdf, reduction_methods)()
    assert_eq(result, expected)


def test_reduction_column_multiindex():
    idx = cudf.MultiIndex.from_tuples(
        [("a", 1), ("a", 2)], names=["foo", "bar"]
    )
    df = cudf.DataFrame(np.array([[1, 3], [2, 4]]), columns=idx)
    result = df.mean()
    expected = df.to_pandas().mean()
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "columns", [pd.RangeIndex(2), pd.Index([0, 1], dtype="int8")]
)
def test_dataframe_axis_0_preserve_column_type_in_index(columns):
    pd_df = pd.DataFrame([[1, 2]], columns=columns)
    cudf_df = cudf.DataFrame.from_pandas(pd_df)
    result = cudf_df.sum(axis=0)
    expected = pd_df.sum(axis=0)
    assert_eq(result, expected, check_index_type=True)


@pytest.mark.parametrize(
    "data_non_overflow",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        [10, 20, 30, None, 100],
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
    if reduction_methods in {
        "var",
        "kurtosis",
        "skew",
        "any",
        "all",
        "product",
    }:
        pytest.skip(
            f"pandas doesn't support {reduction_methods} with {timedelta_types_as_str}"
        )
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
@pytest.mark.parametrize("ddof", [1, 2])
def test_timedelta_std_ddofs(data, timedelta_types_as_str, ddof):
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
