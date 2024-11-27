# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import operator
import re
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_220,
    PANDAS_GT_214,
    PANDAS_VERSION,
)
from cudf.core.dtypes import Decimal32Dtype, Decimal64Dtype, Decimal128Dtype
from cudf.testing import assert_eq
from cudf.testing._utils import (
    INTEGER_TYPES,
    NUMERIC_TYPES,
    assert_exceptions_equal,
    expect_warning_if,
)


@pytest.mark.parametrize(
    "gsr",
    [
        cudf.Series([5, 1, 2, 3, None, 243, None, 4]),
        cudf.Series(["one", "two", "three", None, "one"], dtype="category"),
        cudf.Series([*list(range(400)), None]),
    ],
)
@pytest.mark.parametrize(
    "to_replace,value",
    [
        (0, 5),
        ("one", "two"),
        ("one", "five"),
        ("abc", "hello"),
        ([0, 1], [5, 6]),
        ([22, 323, 27, 0], -1),
        ([1, 2, 3], cudf.Series([10, 11, 12])),
        (cudf.Series([1, 2, 3]), None),
        ({1: 10, 2: 22}, None),
        (np.inf, 4),
    ],
)
def test_series_replace_all(gsr, to_replace, value):
    psr = gsr.to_pandas()

    gd_to_replace = to_replace
    if isinstance(to_replace, cudf.Series):
        pd_to_replace = to_replace.to_pandas()
    else:
        pd_to_replace = to_replace

    gd_value = value
    if isinstance(value, cudf.Series):
        pd_value = value.to_pandas()
    else:
        pd_value = value

    expect_warn = (
        isinstance(gsr.dtype, cudf.CategoricalDtype)
        and isinstance(gd_to_replace, str)
        and gd_to_replace == "one"
    )
    with expect_warning_if(expect_warn):
        actual = gsr.replace(to_replace=gd_to_replace, value=gd_value)
    with expect_warning_if(expect_warn and PANDAS_GE_220):
        if pd_value is None:
            # TODO: Remove this workaround once cudf
            # introduces `no_default` values
            expected = psr.replace(to_replace=pd_to_replace)
        else:
            expected = psr.replace(to_replace=pd_to_replace, value=pd_value)

    assert_eq(
        expected.sort_values().reset_index(drop=True),
        actual.sort_values().reset_index(drop=True),
    )


def test_series_replace():
    a1 = np.array([0, 1, 2, 3, 4])

    # Numerical
    a2 = np.array([5, 1, 2, 3, 4])
    sr1 = cudf.Series(a1)
    sr2 = sr1.replace(0, 5)
    assert_eq(a2, sr2.to_numpy())

    # Categorical
    psr3 = pd.Series(["one", "two", "three"], dtype="category")
    with expect_warning_if(PANDAS_GE_220, FutureWarning):
        psr4 = psr3.replace("one", "two")
    sr3 = cudf.from_pandas(psr3)
    with pytest.warns(FutureWarning):
        sr4 = sr3.replace("one", "two")
    assert_eq(
        psr4.sort_values().reset_index(drop=True),
        sr4.sort_values().reset_index(drop=True),
    )
    with expect_warning_if(PANDAS_GE_220, FutureWarning):
        psr5 = psr3.replace("one", "five")
    with pytest.warns(FutureWarning):
        sr5 = sr3.replace("one", "five")

    assert_eq(psr5, sr5)

    # List input
    a6 = np.array([5, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [5, 6])
    assert_eq(a6, sr6.to_numpy())

    assert_eq(
        sr1.replace([0, 1], [5.5, 6.5]),
        sr1.to_pandas().replace([0, 1], [5.5, 6.5]),
    )

    # Series input
    a8 = np.array([5, 5, 5, 3, 4])
    sr8 = sr1.replace(sr1[:3].to_numpy(), 5)
    assert_eq(a8, sr8.to_numpy())

    # large input containing null
    sr9 = cudf.Series([*list(range(400)), None])
    sr10 = sr9.replace([22, 323, 27, 0], None)
    assert sr10.null_count == 5
    assert len(sr10.dropna().to_numpy()) == (401 - 5)

    sr11 = sr9.replace([22, 323, 27, 0], -1)
    assert sr11.null_count == 1
    assert len(sr11.dropna().to_numpy()) == (401 - 1)

    # large input not containing nulls
    sr9 = sr9.fillna(-11)
    sr12 = sr9.replace([22, 323, 27, 0], None)
    assert sr12.null_count == 4
    assert len(sr12.dropna().to_numpy()) == (401 - 4)

    sr13 = sr9.replace([22, 323, 27, 0], -1)
    assert sr13.null_count == 0
    assert len(sr13.to_numpy()) == 401


def test_series_replace_with_nulls():
    a1 = np.array([0, 1, 2, 3, 4])

    # Numerical
    a2 = np.array([-10, 1, 2, 3, 4])
    sr1 = cudf.Series(a1)
    sr2 = sr1.replace(0, None).fillna(-10)
    assert_eq(a2, sr2.to_numpy())

    # List input
    a6 = np.array([-10, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [None, 6]).fillna(-10)
    assert_eq(a6, sr6.to_numpy())

    sr1 = cudf.Series([0, 1, 2, 3, 4, None])
    assert_eq(
        sr1.replace([0, 1], [5.5, 6.5]).fillna(-10),
        sr1.to_pandas().replace([0, 1], [5.5, 6.5]).fillna(-10),
    )

    # Series input
    a8 = np.array([-10, -10, -10, 3, 4, -10])
    sr8 = sr1.replace(cudf.Series([-10] * 3, index=sr1[:3]), None).fillna(-10)
    assert_eq(a8, sr8.to_numpy())

    a9 = np.array([-10, 6, 2, 3, 4, -10])
    sr9 = sr1.replace([0, 1], [None, 6]).fillna(-10)
    assert_eq(a9, sr9.to_numpy())


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning introduced in pandas-2.2.0",
)
@pytest.mark.parametrize(
    "df",
    [
        cudf.DataFrame(
            {
                "a": [0, 1, None, 2, 3],
                "b": [3, 2, 2, 3, None],
                "c": ["abc", "def", ".", None, None],
            }
        ),
        cudf.DataFrame(
            {
                "a": ["one", "two", None, "three"],
                "b": ["one", None, "two", "three"],
            },
            dtype="category",
        ),
        cudf.DataFrame(
            {
                "col one": [None, 10, 11, None, 1000, 500, 600],
                "col two": ["abc", "def", "ghi", None, "pp", None, "a"],
                "a": [0.324, 0.234, 324.342, 23.32, 9.9, None, None],
            }
        ),
    ],
)
@pytest.mark.parametrize(
    "to_replace,value",
    [
        (0, 4),
        ([0, 1], [4, 5]),
        ([0, 1], 4),
        ({"a": 0, "b": 0}, {"a": 4, "b": 5}),
        ({"a": 0}, {"a": 4}),
        ("abc", "---"),
        ([".", "gh"], "hi"),
        ([".", "def"], ["_", None]),
        ({"c": 0}, {"a": 4, "b": 5}),
        ({"a": 2}, {"c": "a"}),
        ("two", "three"),
        ([1, 2], pd.Series([10, 11])),
        (pd.Series([10, 11], index=[3, 2]), None),
        (
            pd.Series(["a+", "+c", "p", "---"], index=["abc", "gh", "l", "z"]),
            None,
        ),
        (
            pd.Series([10, 11], index=[3, 2]),
            {"a": [-10, -30], "l": [-111, -222]},
        ),
        (pd.Series([10, 11], index=[3, 2]), 555),
        (
            pd.Series([10, 11], index=["a", "b"]),
            pd.Series([555, 1111], index=["a", "b"]),
        ),
        ({"a": "2", "b": "3", "zzz": "hi"}, None),
        ({"a": 2, "b": 3, "zzz": "hi"}, 324353),
        (
            {"a": 2, "b": 3, "zzz": "hi"},
            pd.Series([5, 6, 10], index=["a", "b", "col one"]),
        ),
    ],
)
def test_dataframe_replace(df, to_replace, value):
    gdf = df
    pdf = gdf.to_pandas()

    pd_value = value
    if isinstance(value, pd.Series):
        gd_value = cudf.from_pandas(value)
    else:
        gd_value = value

    pd_to_replace = to_replace
    if isinstance(to_replace, pd.Series):
        gd_to_replace = cudf.from_pandas(to_replace)
    else:
        gd_to_replace = to_replace

    can_warn = (
        isinstance(df["a"].dtype, cudf.CategoricalDtype)
        and isinstance(to_replace, str)
        and to_replace == "two"
        and isinstance(value, str)
        and value == "three"
    )
    with expect_warning_if(can_warn):
        if pd_value is None:
            expected = pdf.replace(to_replace=pd_to_replace)
        else:
            expected = pdf.replace(to_replace=pd_to_replace, value=pd_value)
    with expect_warning_if(can_warn):
        actual = gdf.replace(to_replace=gd_to_replace, value=gd_value)

    expected_sorted = expected.sort_values(by=list(expected.columns), axis=0)
    actual_sorted = actual.sort_values(by=list(actual.columns), axis=0)

    assert_eq(expected_sorted, actual_sorted)


def test_dataframe_replace_with_nulls():
    # numerical
    pdf1 = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]})
    gdf1 = cudf.from_pandas(pdf1)
    pdf2 = pdf1.replace(0, 4)
    gdf2 = gdf1.replace(0, None).fillna(4)
    assert_eq(gdf2, pdf2)

    # list input
    pdf6 = pdf1.replace([0, 1], [4, 5])
    gdf6 = gdf1.replace([0, 1], [4, None]).fillna(5)
    assert_eq(gdf6, pdf6)

    pdf7 = pdf1.replace([0, 1], 4)
    gdf7 = gdf1.replace([0, 1], None).fillna(4)
    assert_eq(gdf7, pdf7)

    # dict input:
    pdf8 = pdf1.replace({"a": 0, "b": 0}, {"a": 4, "b": 5})
    gdf8 = gdf1.replace({"a": 0, "b": 0}, {"a": None, "b": 5}).fillna(4)
    assert_eq(gdf8, pdf8)

    gdf1 = cudf.DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, None]})
    gdf9 = gdf1.replace([0, 1], [4, 5]).fillna(3)
    assert_eq(gdf9, pdf6)


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([0, 1, None, 2, None], dtype=pd.Int8Dtype()),
        pd.Series([0, 1, np.nan, 2, np.nan]),
    ],
)
@pytest.mark.parametrize("data_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("fill_value", [10, pd.Series([10, 20, 30, 40, 50])])
@pytest.mark.parametrize("inplace", [True, False])
def test_series_fillna_numerical(psr, data_dtype, fill_value, inplace):
    test_psr = psr.copy(deep=True)
    # TODO: These tests should use Pandas' nullable int type
    # when we support a recent enough version of Pandas
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
    if np.dtype(data_dtype).kind not in ("f") and test_psr.dtype.kind == "i":
        test_psr = test_psr.astype(
            cudf.utils.dtypes.np_dtypes_to_pandas_dtypes[np.dtype(data_dtype)]
        )

    gsr = cudf.from_pandas(test_psr)

    if isinstance(fill_value, pd.Series):
        fill_value_cudf = cudf.from_pandas(fill_value)
    else:
        fill_value_cudf = fill_value

    expected = test_psr.fillna(fill_value, inplace=inplace)
    actual = gsr.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        expected = test_psr
        actual = gsr

    # TODO: Remove check_dtype when we have support
    # to compare with pandas nullable dtypes
    assert_eq(expected, actual, check_dtype=False)


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
@pytest.mark.parametrize("data_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("method", ["ffill", "bfill"])
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_method_numerical(data, container, data_dtype, method, inplace):
    if container == pd.DataFrame:
        data = {"a": data, "b": data, "c": data}

    pdata = container(data)

    if np.dtype(data_dtype).kind not in ("f"):
        data_dtype = cudf.utils.dtypes.np_dtypes_to_pandas_dtypes[
            np.dtype(data_dtype)
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
    "gsr_data",
    [
        cudf.Series(["2.34", "5.2", "7.47", None, "92.29", None]).astype(
            Decimal64Dtype(7, 2)
        ),
        cudf.Series(["-74.56", None, "-23.73", "34.55", "2.89", None]).astype(
            Decimal32Dtype(7, 2)
        ),
        cudf.Series(
            ["85.955", np.nan, "-3.243", np.nan, "29.492", np.nan]
        ).astype(Decimal64Dtype(8, 3)),
        cudf.Series(
            ["2.964", None, "57.432", "-989.330", None, "56.444"]
        ).astype(Decimal64Dtype(8, 3)),
        cudf.Series(
            [np.nan, "55.2498", np.nan, "-5.2965", "-28.9423", np.nan]
        ).astype(Decimal64Dtype(10, 4)),
        cudf.Series(
            ["2.964", None, "54347.432", "-989.330", None, "56.444"]
        ).astype(Decimal128Dtype(20, 7)),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        42,
        -123,
        Decimal("8.2"),
        Decimal("-12.87"),
        cudf.Series([None, -854, 9533, -274, -845, 7924], dtype="int32"),
        cudf.Series(["-53.5", "13.4", "-64.3", None, "42.42", None]).astype(
            Decimal64Dtype(7, 2)
        ),
        cudf.Series(
            ["57.45", np.nan, np.nan, "686.49", "-55.5", "73.24"],
        ).astype(Decimal64Dtype(7, 2)),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_decimal(gsr_data, fill_value, inplace):
    gsr = gsr_data.copy(deep=True)
    psr = gsr.to_pandas()

    if isinstance(fill_value, cudf.Series):
        p_fill_value = fill_value.to_pandas()
    else:
        p_fill_value = fill_value

    expected = psr.fillna(p_fill_value, inplace=inplace)
    got = gsr.fillna(fill_value, inplace=inplace)

    assert_eq(expected, got, check_dtype=False)


@pytest.mark.parametrize(
    "psr_data",
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
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_categorical(psr_data, fill_value, inplace):
    psr = psr_data.copy(deep=True)
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
    "psr_data",
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
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_datetime(psr_data, fill_value, inplace):
    psr = psr_data.copy(deep=True)
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
@pytest.mark.parametrize("inplace", [True, False])
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
    "df",
    [
        pd.DataFrame({"a": [1, 2, None], "b": [None, None, 5]}),
        pd.DataFrame(
            {"a": [1, 2, None], "b": [None, None, 5]}, index=["a", "p", "z"]
        ),
        pd.DataFrame({"a": [1, 2, 3]}),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        10,
        pd.Series([10, 20, 30]),
        pd.Series([3, 4, 5]),
        pd.Series([10, 20, 30], index=["z", "a", "p"]),
        {"a": 5, "b": pd.Series([3, 4, 5])},
        {"a": 5001},
        {"b": pd.Series([11, 22, 33], index=["a", "p", "z"])},
        {"a": 5, "b": pd.Series([3, 4, 5], index=["a", "p", "z"])},
        {"c": 100},
        np.nan,
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_dataframe(df, value, inplace):
    pdf = df.copy(deep=True)
    gdf = cudf.from_pandas(pdf)

    fill_value_pd = value
    if isinstance(fill_value_pd, (pd.Series, pd.DataFrame)):
        fill_value_cudf = cudf.from_pandas(fill_value_pd)
    elif isinstance(fill_value_pd, dict):
        fill_value_cudf = {}
        for key in fill_value_pd:
            temp_val = fill_value_pd[key]
            if isinstance(temp_val, pd.Series):
                temp_val = cudf.from_pandas(temp_val)
            fill_value_cudf[key] = temp_val
    else:
        fill_value_cudf = value

    expect = pdf.fillna(fill_value_pd, inplace=inplace)
    got = gdf.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        got = gdf
        expect = pdf

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "ps_data",
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
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_string(ps_data, fill_value, inplace):
    psr = ps_data.copy(deep=True)
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


@pytest.mark.parametrize("data_dtype", INTEGER_TYPES)
def test_series_fillna_invalid_dtype(data_dtype):
    gdf = cudf.Series([1, 2, None, 3], dtype=data_dtype)
    fill_value = 2.5
    msg = (
        f"Cannot safely cast non-equivalent"
        f" {type(fill_value).__name__} to {gdf.dtype.type.__name__}"
    )
    with pytest.raises(TypeError, match=msg):
        gdf.fillna(fill_value)


@pytest.mark.parametrize("data_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("fill_value", [100, 100.0, 128.5])
@pytest.mark.parametrize("op", [operator.gt, operator.eq, operator.lt])
def test_series_where(data_dtype, fill_value, op):
    psr = pd.Series(list(range(10)), dtype=data_dtype)
    sr = cudf.from_pandas(psr)

    try:
        scalar_fits = sr.dtype.type(fill_value) == fill_value
    except OverflowError:
        scalar_fits = False

    if not scalar_fits:
        with pytest.raises(TypeError):
            sr.where(op(sr, 0), fill_value)
    else:
        # Cast back to original dtype as pandas automatically upcasts
        expect = psr.where(op(psr, 0), fill_value)
        got = sr.where(op(sr, 0), fill_value)
        # pandas returns 'float16' dtype, which is not supported in cudf
        assert_eq(
            expect,
            got,
            check_dtype=expect.dtype.kind not in ("f"),
        )


@pytest.mark.parametrize("fill_value", [100, 100.0, 100.5])
def test_series_with_nulls_where(fill_value):
    psr = pd.Series([None] * 3 + list(range(5)))
    sr = cudf.from_pandas(psr)

    expect = psr.where(psr > 0, fill_value)
    got = sr.where(sr > 0, fill_value)
    assert_eq(expect, got)

    expect = psr.where(psr < 0, fill_value)
    got = sr.where(sr < 0, fill_value)
    assert_eq(expect, got)

    expect = psr.where(psr == 0, fill_value)
    got = sr.where(sr == 0, fill_value)
    assert_eq(expect, got)


@pytest.mark.parametrize("fill_value", [[888, 999]])
def test_dataframe_with_nulls_where_with_scalars(fill_value):
    pdf = pd.DataFrame(
        {
            "A": [-1, 2, -3, None, 5, 6, -7, 0],
            "B": [4, -2, 3, None, 7, 6, 8, 0],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.where(pdf % 3 == 0, fill_value)
    got = gdf.where(gdf % 3 == 0, fill_value)

    assert_eq(expect, got)


def test_dataframe_with_different_types():
    # Testing for int and float
    pdf = pd.DataFrame(
        {"A": [111, 22, 31, 410, 56], "B": [-10.12, 121.2, 45.7, 98.4, 87.6]}
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.where(pdf > 50, -pdf)
    got = gdf.where(gdf > 50, -gdf)

    assert_eq(expect, got)

    # Testing for string
    pdf = pd.DataFrame({"A": ["a", "bc", "cde", "fghi"]})
    gdf = cudf.from_pandas(pdf)
    pdf_mask = pd.DataFrame({"A": [True, False, True, False]})
    gdf_mask = cudf.from_pandas(pdf_mask)
    expect = pdf.where(pdf_mask, ["cudf"])
    got = gdf.where(gdf_mask, ["cudf"])

    assert_eq(expect, got)

    # Testing for categoriacal
    pdf = pd.DataFrame({"A": ["a", "b", "b", "c"]})
    pdf["A"] = pdf["A"].astype("category")
    gdf = cudf.from_pandas(pdf)
    expect = pdf.where(pdf_mask, "c")
    got = gdf.where(gdf_mask, ["c"])

    assert_eq(expect, got)


def test_dataframe_where_with_different_options():
    pdf = pd.DataFrame({"A": [1, 2, 3], "B": [3, 4, 5]})
    gdf = cudf.from_pandas(pdf)

    # numpy array
    boolean_mask = np.array([[False, True], [True, False], [False, True]])

    expect = pdf.where(boolean_mask, -pdf)
    got = gdf.where(boolean_mask, -gdf)

    assert_eq(expect, got)

    # with single scalar
    expect = pdf.where(boolean_mask, 8)
    got = gdf.where(boolean_mask, 8)

    assert_eq(expect, got)

    # with multi scalar
    expect = pdf.where(boolean_mask, [8, 9])
    got = gdf.where(boolean_mask, [8, 9])

    assert_eq(expect, got)


def test_series_multiple_times_with_nulls():
    sr = cudf.Series([1, 2, 3, None])
    expected = cudf.Series([None, None, None, None], dtype=np.int64)

    for i in range(3):
        got = sr.replace([1, 2, 3], None)
        assert_eq(expected, got)
        # BUG: #2695
        # The following series will acquire a chunk of memory and update with
        # values, but these values may still linger even after the memory
        # gets released. This memory space might get used for replace in
        # subsequent calls and the memory used for mask may have junk values.
        # So, if it is not updated properly, the result would be wrong.
        # So, this will help verify that scenario.
        cudf.Series([1, 1, 1, None])


@pytest.mark.parametrize("series_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize(
    "replacement", [128, 128.0, 128.5, 32769, 32769.0, 32769.5]
)
def test_numeric_series_replace_dtype(request, series_dtype, replacement):
    request.applymarker(
        pytest.mark.xfail(
            condition=PANDAS_GT_214
            and (
                (
                    series_dtype == "int8"
                    and replacement in {128, 128.0, 32769, 32769.0}
                )
                or (
                    series_dtype == "int16" and replacement in {32769, 32769.0}
                )
            ),
            reason="Pandas throws an AssertionError for these "
            "cases and asks us to log a bug, they are trying to "
            "avoid a RecursionError which cudf will not run into",
        )
    )
    psr = pd.Series([0, 1, 2, 3, 4, 5], dtype=series_dtype)
    sr = cudf.from_pandas(psr)

    expect = psr.replace(1, replacement)
    got = sr.replace(1, replacement)

    assert_eq(expect, got)

    # to_replace is a list, replacement is a scalar
    expect = psr.replace([2, 3], replacement)
    got = sr.replace([2, 3], replacement)

    assert_eq(expect, got)

    # If to_replace is a scalar and replacement is a list
    with pytest.raises(TypeError):
        sr.replace(0, [replacement, 2])

    # Both list of unequal length
    with pytest.raises(ValueError):
        sr.replace([0, 1], [replacement])

    # Both lists of equal length
    expect = psr.replace([2, 3], [replacement, replacement])
    got = sr.replace([2, 3], [replacement, replacement])
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "pframe, replace_args",
    [
        (
            pd.Series([5, 1, 2, 3, 4]),
            {"to_replace": 5, "value": 0, "inplace": True},
        ),
        (
            pd.Series([5, 1, 2, 3, 4]),
            {"to_replace": {5: 0, 3: -5}, "inplace": True},
        ),
        (pd.Series([5, 1, 2, 3, 4]), {}),
        pytest.param(
            pd.Series(["one", "two", "three"], dtype="category"),
            {"to_replace": "one", "value": "two", "inplace": True},
            marks=pytest.mark.xfail(
                condition=PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION,
                reason="https://github.com/pandas-dev/pandas/issues/43232"
                "https://github.com/pandas-dev/pandas/issues/53358",
            ),
        ),
        (
            pd.DataFrame({"A": [0, 1, 2, 3, 4], "B": [5, 6, 7, 8, 9]}),
            {"to_replace": 5, "value": 0, "inplace": True},
        ),
        (
            pd.Series([1, 2, 3, 45]),
            {
                "to_replace": np.array([]).astype(int),
                "value": 77,
                "inplace": True,
            },
        ),
        (
            pd.Series([1, 2, 3, 45]),
            {
                "to_replace": np.array([]).astype(int),
                "value": 77,
                "inplace": False,
            },
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]}),
            {"to_replace": {"a": 2}, "value": {"a": -33}, "inplace": True},
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]}),
            {
                "to_replace": {"a": [2, 5]},
                "value": {"a": [9, 10]},
                "inplace": True,
            },
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]}),
            {"to_replace": [], "value": [], "inplace": True},
        ),
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warning not given on older versions of pandas",
)
def test_replace_inplace(pframe, replace_args):
    gpu_frame = cudf.from_pandas(pframe)
    pandas_frame = pframe.copy()

    gpu_copy = gpu_frame.copy()
    cpu_copy = pandas_frame.copy()

    assert_eq(gpu_frame, pandas_frame)
    assert_eq(gpu_copy, cpu_copy)
    with expect_warning_if(len(replace_args) == 0):
        gpu_frame.replace(**replace_args)
    with expect_warning_if(len(replace_args) == 0):
        pandas_frame.replace(**replace_args)
    assert_eq(gpu_frame, pandas_frame)
    assert_eq(gpu_copy, cpu_copy)


def test_replace_df_error():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]})
    gdf = cudf.from_pandas(pdf)

    assert_exceptions_equal(
        lfunc=pdf.replace,
        rfunc=gdf.replace,
        lfunc_args_and_kwargs=([], {"to_replace": -1, "value": []}),
        rfunc_args_and_kwargs=([], {"to_replace": -1, "value": []}),
    )


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([2, 7.4], [4, 7.9]),
        ([2, 7.4], None),
        (
            None,
            [4, 7.9],
        ),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_clip(lower, upper, inplace):
    pdf = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [7.1, 7.24, 7.5, 7.8, 8.11]}
    )
    gdf = cudf.from_pandas(pdf)

    got = gdf.clip(lower=lower, upper=upper, inplace=inplace)
    expect = pdf.clip(lower=lower, upper=upper, axis=1)

    if inplace is True:
        assert_eq(expect, gdf)
    else:
        assert_eq(expect, got)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [("b", "d"), ("b", None), (None, "c"), (None, None)],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_category_clip(lower, upper, inplace):
    data = ["a", "b", "c", "d", "e"]
    pdf = pd.DataFrame({"a": data})
    gdf = cudf.from_pandas(pdf)
    gdf["a"] = gdf["a"].astype("category")

    expect = pdf.clip(lower=lower, upper=upper)
    got = gdf.clip(lower=lower, upper=upper, inplace=inplace)

    if inplace is True:
        assert_eq(expect, gdf.astype("str"))
    else:
        assert_eq(expect, got.astype("str"))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [([2, 7.4], [4, 7.9, "d"]), ([2, 7.4, "a"], [4, 7.9, "d"])],
)
def test_dataframe_exceptions_for_clip(lower, upper):
    gdf = cudf.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [7.1, 7.24, 7.5, 7.8, 8.11]}
    )

    with pytest.raises(ValueError):
        gdf.clip(lower=lower, upper=upper)


@pytest.mark.parametrize(
    ("data", "lower", "upper"),
    [
        ([1, 2, 3, 4, 5], 2, 4),
        ([1, 2, 3, 4, 5], 2, None),
        ([1, 2, 3, 4, 5], None, 4),
        ([1, 2, 3, 4, 5], None, None),
        ([1, 2, 3, 4, 5], 4, 2),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 4, 2),
        (pd.Series([1, 2, 3, 4, 5], dtype="int32"), 4, 2),
        (["a", "b", "c", "d", "e"], "b", "d"),
        (["a", "b", "c", "d", "e"], "b", None),
        (["a", "b", "c", "d", "e"], None, "d"),
        (["a", "b", "c", "d", "e"], "d", "b"),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_series_clip(data, lower, upper, inplace):
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)

    expect = psr.clip(lower=lower, upper=upper)
    got = gsr.clip(lower=lower, upper=upper, inplace=inplace)

    if inplace is True:
        assert_eq(expect, gsr)
    else:
        assert_eq(expect, got)


def test_series_exceptions_for_clip():
    with pytest.raises(ValueError):
        cudf.Series([1, 2, 3, 4]).clip([1, 2], [2, 3])

    with pytest.raises(NotImplementedError):
        cudf.Series([1, 2, 3, 4]).clip(1, 2, axis=0)


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


def test_series_replace_errors():
    gsr = cudf.Series([1, 2, None, 3, None])
    psr = gsr.to_pandas()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "to_replace and value should be of same types,"
            "got to_replace dtype: int64 and "
            "value dtype: object"
        ),
    ):
        gsr.replace(1, "a")

    gsr = cudf.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "to_replace and value should be of same types,"
            "got to_replace dtype: int64 and "
            "value dtype: object"
        ),
    ):
        gsr.replace([1, 2], ["a", "b"])

    assert_exceptions_equal(
        psr.replace,
        gsr.replace,
        ([{"a": 1}, 1],),
        ([{"a": 1}, 1],),
    )

    assert_exceptions_equal(
        lfunc=psr.replace,
        rfunc=gsr.replace,
        lfunc_args_and_kwargs=([[1, 2], [1]],),
        rfunc_args_and_kwargs=([[1, 2], [1]],),
    )

    assert_exceptions_equal(
        lfunc=psr.replace,
        rfunc=gsr.replace,
        lfunc_args_and_kwargs=([object(), [1]],),
        rfunc_args_and_kwargs=([object(), [1]],),
    )

    assert_exceptions_equal(
        lfunc=psr.replace,
        rfunc=gsr.replace,
        lfunc_args_and_kwargs=([{"a": 1}, object()],),
        rfunc_args_and_kwargs=([{"a": 1}, object()],),
    )


@pytest.mark.parametrize(
    "gsr,old,new,expected",
    [
        (
            cudf.Series(["a", "b", "c", None]),
            None,
            "a",
            cudf.Series(["a", "b", "c", "a"]),
        ),
        (
            cudf.Series(["a", "b", "c", None]),
            [None, "a", "a"],
            ["c", "b", "d"],
            cudf.Series(["d", "b", "c", "c"]),
        ),
        (
            cudf.Series(["a", "b", "c", None]),
            [None, "a"],
            ["b", None],
            cudf.Series([None, "b", "c", "b"]),
        ),
        (
            cudf.Series(["a", "b", "c", None]),
            [None, None],
            [None, None],
            cudf.Series(["a", "b", "c", None]),
        ),
        (cudf.Series([1, 2, None, 3]), None, 10, cudf.Series([1, 2, 10, 3])),
        (
            cudf.Series([1, 2, None, 3]),
            [None, 1, 1],
            [3, 2, 4],
            cudf.Series([4, 2, 3, 3]),
        ),
        (
            cudf.Series([1, 2, None, 3]),
            [None, 1],
            [2, None],
            cudf.Series([None, 2, 2, 3]),
        ),
        (
            cudf.Series(["a", "q", "t", None], dtype="category"),
            None,
            "z",
            cudf.Series(["a", "q", "t", "z"], dtype="category"),
        ),
        (
            cudf.Series(["a", "q", "t", None], dtype="category"),
            [None, "a", "q"],
            ["z", None, None],
            cudf.Series([None, None, "t", "z"], dtype="category"),
        ),
        (
            cudf.Series(["a", None, "t", None], dtype="category"),
            [None, "t"],
            ["p", None],
            cudf.Series(["a", "p", None, "p"], dtype="category"),
        ),
    ],
)
def test_replace_nulls(gsr, old, new, expected):
    with expect_warning_if(isinstance(gsr.dtype, cudf.CategoricalDtype)):
        actual = gsr.replace(old, new)
    assert_eq(
        expected.sort_values().reset_index(drop=True),
        actual.sort_values().reset_index(drop=True),
    )


def test_fillna_columns_multiindex():
    columns = pd.MultiIndex.from_tuples([("a", "b"), ("d", "e")])
    pdf = pd.DataFrame(
        {"0": [1, 2, None, 3, None], "1": [None, None, None, None, 4]}
    )
    pdf.columns = columns
    gdf = cudf.from_pandas(pdf)

    expected = pdf.fillna(10)
    actual = gdf.fillna(10)

    assert_eq(expected, actual)


def test_fillna_nan_and_null():
    ser = cudf.Series(pa.array([float("nan"), None, 1.1]), nan_as_null=False)
    result = ser.fillna(2.2)
    expected = cudf.Series([2.2, 2.2, 1.1])
    assert_eq(result, expected)


def test_replace_with_index_objects():
    result = cudf.Series([1, 2]).replace(cudf.Index([1]), cudf.Index([2]))
    expected = pd.Series([1, 2]).replace(pd.Index([1]), pd.Index([2]))
    assert_eq(result, expected)


# Example test function for datetime series replace
def test_replace_datetime_series():
    # Create a pandas datetime series
    pd_series = pd.Series(pd.date_range("20210101", periods=5))
    # Replace a specific datetime value
    pd_result = pd_series.replace(
        pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-10")
    )

    # Create a cudf datetime series
    cudf_series = cudf.Series(pd.date_range("20210101", periods=5))
    # Replace a specific datetime value
    cudf_result = cudf_series.replace(
        pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-10")
    )

    assert_eq(pd_result, cudf_result)


# Example test function for timedelta series replace
def test_replace_timedelta_series():
    # Create a pandas timedelta series
    pd_series = pd.Series(pd.timedelta_range("1 days", periods=5))
    # Replace a specific timedelta value
    pd_result = pd_series.replace(
        pd.Timedelta("2 days"), pd.Timedelta("10 days")
    )

    # Create a cudf timedelta series
    cudf_series = cudf.Series(pd.timedelta_range("1 days", periods=5))
    # Replace a specific timedelta value
    cudf_result = cudf_series.replace(
        pd.Timedelta("2 days"), pd.Timedelta("10 days")
    )

    assert_eq(pd_result, cudf_result)


def test_replace_multiple_rows(datadir):
    path = datadir / "parquet" / "replace_multiple_rows.parquet"
    pdf = pd.read_parquet(path)
    gdf = cudf.read_parquet(path)

    pdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    gdf.replace([np.inf, -np.inf], np.nan, inplace=True)

    assert_eq(pdf, gdf, check_dtype=False)
