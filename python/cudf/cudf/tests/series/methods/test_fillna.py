# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_220,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "data",
    [
        [1, None, 11, 2.0, np.nan],
        [np.nan],
        [None, None, None],
        [np.nan, 1, 10, 393.32, np.nan],
    ],
)
@pytest.mark.parametrize("fill_value", [1.2, 332, np.nan])
def test_fillna_with_nan(data, nan_as_null, fill_value):
    gs = cudf.Series(data, dtype="float64", nan_as_null=nan_as_null)
    ps = gs.to_pandas()

    expected = ps.fillna(fill_value)
    actual = gs.fillna(fill_value)

    assert_eq(expected, actual)


def test_fillna_categorical_with_non_categorical_raises():
    ser = cudf.Series([1, None], dtype="category")
    with pytest.raises(TypeError):
        ser.fillna(cudf.Series([1, 2]))


def test_fillna_categorical_with_different_categories_raises():
    ser = cudf.Series([1, None], dtype="category")
    with pytest.raises(TypeError):
        ser.fillna(cudf.Series([1, 2]), dtype="category")


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2],
        [None, 1],
        [None, None],
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        np.datetime64("2005-02"),
        np.datetime64("2005-02-25"),
        np.datetime64("2005-02-25T03:30"),
        np.datetime64("nat"),
        "NaT",
    ],
)
def test_datetime_fillna(data, datetime_types_as_str, fill_value):
    sr = cudf.Series(data, dtype=datetime_types_as_str)
    psr = sr.to_pandas()

    expected = psr.dropna()
    actual = sr.dropna()

    assert_eq(expected, actual)

    expected = psr.fillna(fill_value)
    actual = sr.fillna(fill_value)

    assert_eq(expected, actual)

    expected = expected.dropna()
    actual = actual.dropna()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [12, 12, 22, 343, 4353534, 435342],
        [0.3534, 12, 22, 343, 43.53534, 4353.42],
        np.array([10, 20, 30, 100]),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        np.timedelta64(4, "s"),
        np.timedelta64(456, "D"),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
        "NaT",
    ],
)
def test_timedelta_fillna(data, timedelta_types_as_str, fill_value):
    sr = cudf.Series(data, dtype=timedelta_types_as_str)
    psr = sr.to_pandas()

    expected = psr.dropna()
    actual = sr.dropna()

    assert_eq(expected, actual)

    expected = psr.fillna(fill_value)
    actual = sr.fillna(fill_value)
    assert_eq(expected, actual)

    expected = expected.dropna()
    actual = actual.dropna()

    assert_eq(expected, actual)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "data",
    [
        [1, None, None, 2, 3, 4],
        [None, None, 1, 2, None, 3, 4],
        [1, 2, None, 3, 4, None, None],
        [0] + [None] * 14,
        [None] * 14 + [0],
    ],
)
@pytest.mark.parametrize("container", [pd.Series, pd.DataFrame])
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_fillna_method_numerical(
    data, container, numeric_types_as_str, method, inplace
):
    if container == pd.DataFrame:
        data = {"a": data, "b": data, "c": data}

    pdata = container(data)

    data_dtype = numeric_types_as_str
    if np.dtype(numeric_types_as_str).kind != "f":
        data_dtype = cudf.utils.dtypes.np_dtypes_to_pandas_dtypes[
            np.dtype(numeric_types_as_str)
        ]
    pdata = pdata.astype(data_dtype)

    # Explicitly using nans_as_nulls=True
    gdata = cudf.from_pandas(pdata, nan_as_null=True)

    with pytest.warns(FutureWarning):
        expected = pdata.fillna(method=method, inplace=inplace)
    with pytest.warns(FutureWarning):
        actual = gdata.fillna(method=method, inplace=inplace)

    if inplace:
        expected = pdata
        actual = gdata

    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.parametrize(
    "gsr_data, dtype",
    [
        (
            ["2.34", "5.2", "7.47", None, "92.29", None],
            cudf.Decimal64Dtype(7, 2),
        ),
        (
            ["-74.56", None, "-23.73", "34.55", "2.89", None],
            cudf.Decimal32Dtype(7, 2),
        ),
        (
            ["85.955", np.nan, "-3.243", np.nan, "29.492", np.nan],
            cudf.Decimal64Dtype(8, 3),
        ),
        (
            ["2.964", None, "57.432", "-989.330", None, "56.444"],
            cudf.Decimal64Dtype(8, 3),
        ),
        (
            [np.nan, "55.2498", np.nan, "-5.2965", "-28.9423", np.nan],
            cudf.Decimal64Dtype(10, 4),
        ),
        (
            ["2.964", None, "54347.432", "-989.330", None, "56.444"],
            cudf.Decimal128Dtype(20, 7),
        ),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        lambda: 42,
        lambda: -123,
        lambda: decimal.Decimal("8.2"),
        lambda: decimal.Decimal("-12.87"),
        lambda: cudf.Series(
            [None, -854, 9533, -274, -845, 7924], dtype="int32"
        ),
        lambda: cudf.Series(
            ["-53.5", "13.4", "-64.3", None, "42.42", None]
        ).astype(cudf.Decimal64Dtype(7, 2)),
        lambda: cudf.Series(
            ["57.45", np.nan, np.nan, "686.49", "-55.5", "73.24"],
        ).astype(cudf.Decimal64Dtype(7, 2)),
    ],
)
def test_fillna_decimal(gsr_data, dtype, fill_value, inplace):
    gsr = cudf.Series(gsr_data).astype(dtype)
    psr = gsr.to_pandas()
    fill_value = fill_value()
    if isinstance(fill_value, cudf.Series):
        p_fill_value = fill_value.to_pandas()
    else:
        p_fill_value = fill_value

    expected = psr.fillna(p_fill_value, inplace=inplace)
    got = gsr.fillna(fill_value, inplace=inplace)

    assert_eq(expected, got, check_dtype=False)


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series(["a", "b", "a", None, "c", None], dtype="category"),
        pd.Series(
            ["a", "b", "a", None, "c", None],
            dtype="category",
            index=["q", "r", "z", "a", "b", "c"],
        ),
        pd.Series(
            ["a", "b", "a", None, "c", None],
            dtype="category",
            index=["x", "t", "p", "q", "r", "z"],
        ),
        pd.Series(["a", "b", "a", np.nan, "c", np.nan], dtype="category"),
        pd.Series(
            [None, None, None, None, None, None, "a", "b", "c"],
            dtype="category",
        ),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        "c",
        pd.Series(["c", "c", "c", "c", "c", "a"], dtype="category"),
        pd.Series(
            ["a", "b", "a", None, "c", None],
            dtype="category",
            index=["x", "t", "p", "q", "r", "z"],
        ),
        pd.Series(
            ["a", "b", "a", None, "c", None],
            dtype="category",
            index=["q", "r", "z", "a", "b", "c"],
        ),
        pd.Series(["a", "b", "a", None, "c", None], dtype="category"),
        pd.Series(["a", "b", "a", np.nan, "c", np.nan], dtype="category"),
    ],
)
def test_fillna_categorical(psr, fill_value, inplace):
    if inplace:
        psr = psr.copy(deep=True)
    gsr = cudf.from_pandas(psr)

    if isinstance(fill_value, pd.Series):
        fill_value_cudf = cudf.from_pandas(fill_value)
    else:
        fill_value_cudf = fill_value

    if (
        isinstance(fill_value_cudf, cudf.Series)
        and gsr.dtype != fill_value_cudf.dtype
    ):
        assert_exceptions_equal(
            lfunc=psr.fillna,
            rfunc=gsr.fillna,
            lfunc_args_and_kwargs=([fill_value], {"inplace": inplace}),
            rfunc_args_and_kwargs=([fill_value_cudf], {"inplace": inplace}),
        )
    else:
        expected = psr.fillna(fill_value, inplace=inplace)
        got = gsr.fillna(fill_value_cudf, inplace=inplace)

        if inplace:
            expected = psr
            got = gsr

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series(
            pd.date_range(
                "2010-01-01",
                "2020-01-10",
                freq="1YE" if PANDAS_GE_220 else "1y",
            )
        ),
        pd.Series(["2010-01-01", None, "2011-10-10"], dtype="datetime64[ns]"),
        pd.Series(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                "2011-10-10",
                "2010-01-01",
                "2010-01-02",
                "2010-01-04",
                "2010-11-01",
            ],
            dtype="datetime64[ns]",
        ),
        pd.Series(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                "2011-10-10",
                "2010-01-01",
                "2010-01-02",
                "2010-01-04",
                "2010-11-01",
            ],
            dtype="datetime64[ns]",
            index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        ),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        pd.Timestamp("2010-01-02"),
        pd.Series(
            pd.date_range(
                "2010-01-01",
                "2020-01-10",
                freq="1YE" if PANDAS_GE_220 else "1y",
            )
        )
        + pd.Timedelta("1d"),
        pd.Series(["2010-01-01", None, "2011-10-10"], dtype="datetime64[ns]"),
        pd.Series(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                "2011-10-10",
                "2010-01-01",
                "2010-01-02",
                "2010-01-04",
                "2010-11-01",
            ],
            dtype="datetime64[ns]",
        ),
        pd.Series(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                "2011-10-10",
                "2010-01-01",
                "2010-01-02",
                "2010-01-04",
                "2010-11-01",
            ],
            dtype="datetime64[ns]",
            index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        ),
    ],
)
def test_fillna_datetime(psr, fill_value, inplace):
    if inplace:
        psr = psr.copy(deep=True)
    gsr = cudf.from_pandas(psr)

    if isinstance(fill_value, pd.Series):
        fill_value_cudf = cudf.from_pandas(fill_value)
    else:
        fill_value_cudf = fill_value

    expected = psr.fillna(fill_value, inplace=inplace)
    got = gsr.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        got = gsr
        expected = psr

    assert_eq(expected, got)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "data",
    [
        # Categorical
        pd.Categorical([1, 2, None, None, 3, 4]),
        pd.Categorical([None, None, 1, None, 3, 4]),
        pd.Categorical([1, 2, None, 3, 4, None, None]),
        pd.Categorical(["1", "20", None, None, "3", "40"]),
        pd.Categorical([None, None, "10", None, "30", "4"]),
        pd.Categorical(["1", "20", None, "30", "4", None, None]),
        # Datetime
        np.array(
            [
                "2020-01-01 08:00:00",
                "2020-01-01 09:00:00",
                None,
                "2020-01-01 10:00:00",
                None,
                "2020-01-01 10:00:00",
            ],
            dtype="datetime64[ns]",
        ),
        np.array(
            [
                None,
                None,
                "2020-01-01 09:00:00",
                "2020-01-01 10:00:00",
                None,
                "2020-01-01 10:00:00",
            ],
            dtype="datetime64[ns]",
        ),
        np.array(
            [
                "2020-01-01 09:00:00",
                None,
                None,
                "2020-01-01 10:00:00",
                None,
                None,
            ],
            dtype="datetime64[ns]",
        ),
        # Timedelta
        np.array(
            [10, 100, 1000, None, None, 10, 100, 1000], dtype="datetime64[ns]"
        ),
        np.array(
            [None, None, 10, None, 1000, 100, 10], dtype="datetime64[ns]"
        ),
        np.array(
            [10, 100, None, None, 1000, None, None], dtype="datetime64[ns]"
        ),
        # String
        np.array(
            ["10", "100", "1000", None, None, "10", "100", "1000"],
            dtype="object",
        ),
        np.array(
            [None, None, "1000", None, "10", "100", "10"], dtype="object"
        ),
        np.array(
            ["10", "100", None, None, "1000", None, None], dtype="object"
        ),
    ],
)
@pytest.mark.parametrize("container", [pd.Series, pd.DataFrame])
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_fillna_method_fixed_width_non_num(data, container, method, inplace):
    if container == pd.DataFrame:
        data = {"a": data, "b": data, "c": data}

    pdata = container(data)

    # Explicitly using nans_as_nulls=True
    gdata = cudf.from_pandas(pdata, nan_as_null=True)

    with pytest.warns(FutureWarning):
        expected = pdata.fillna(method=method, inplace=inplace)
    with pytest.warns(FutureWarning):
        actual = gdata.fillna(method=method, inplace=inplace)

    if inplace:
        expected = pdata
        actual = gdata

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series(["a", "b", "c", "d"]),
        pd.Series([None] * 4, dtype="object"),
        pd.Series(["z", None, "z", None]),
        pd.Series(["x", "y", None, None, None]),
        pd.Series([None, None, None, "i", "P"]),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        "a",
        pd.Series(["a", "b", "c", "d"]),
        pd.Series(["z", None, "z", None]),
        pd.Series([None] * 4, dtype="object"),
        pd.Series(["x", "y", None, None, None]),
        pd.Series([None, None, None, "i", "P"]),
    ],
)
def test_fillna_string(psr, fill_value, inplace):
    if inplace:
        psr = psr.copy(deep=True)
    gsr = cudf.from_pandas(psr)

    if isinstance(fill_value, pd.Series):
        fill_value_cudf = cudf.from_pandas(fill_value)
    else:
        fill_value_cudf = fill_value

    expected = psr.fillna(fill_value, inplace=inplace)
    got = gsr.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        expected = psr
        got = gsr

    assert_eq(expected, got)


def test_series_fillna_invalid_dtype(integer_types_as_str):
    gdf = cudf.Series([1, 2, None, 3], dtype=integer_types_as_str)
    fill_value = 2.5
    msg = (
        f"Cannot safely cast non-equivalent"
        f" {type(fill_value).__name__} to {gdf.dtype.type.__name__}"
    )
    with pytest.raises(TypeError, match=msg):
        gdf.fillna(fill_value)


@pytest.mark.parametrize(
    "data", [[1, 2.0, 3, 4, None, 1, None, 10, None], ["a", "b", "c"]]
)
@pytest.mark.parametrize(
    "index",
    [
        None,
        [1, 2, 3],
        ["a", "b", "z"],
        ["a", "b", "c", "d", "e", "f", "g", "l", "m"],
    ],
)
@pytest.mark.parametrize("value", [[1, 2, 3, 4, None, 1, None, 10, None]])
def test_series_fillna(data, index, value):
    psr = pd.Series(
        data,
        index=index if index is not None and len(index) == len(data) else None,
    )
    gsr = cudf.Series(
        data,
        index=index if index is not None and len(index) == len(data) else None,
    )

    expect = psr.fillna(pd.Series(value))
    got = gsr.fillna(cudf.Series(value))
    assert_eq(expect, got)


def test_series_fillna_error():
    psr = pd.Series([1, 2, None, 3, None])
    gsr = cudf.from_pandas(psr)

    assert_exceptions_equal(
        psr.fillna,
        gsr.fillna,
        ([pd.DataFrame({"a": [1, 2, 3]})],),
        ([cudf.DataFrame({"a": [1, 2, 3]})],),
    )


def test_fillna_nan_and_null():
    ser = cudf.Series(pa.array([float("nan"), None, 1.1]), nan_as_null=False)
    result = ser.fillna(2.2)
    expected = cudf.Series([2.2, 2.2, 1.1])
    assert_eq(result, expected)
